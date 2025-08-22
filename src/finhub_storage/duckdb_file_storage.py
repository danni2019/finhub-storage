import duckdb
import os
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from infra.data.local_file_storage import LocalFileStorage
from infra.tool.log import log
from infra.tool.config import file_storage_config, project_path

logger = log(__file__, "infra")

# NOTE:
# This class is designed to be dropped alongside your existing LocalFileStorage.
# It subclasses LocalFileStorage and only overrides the *necessary* methods while
# keeping public method names & signatures unchanged.
#
# Key improvements:
# - insert_dataframe: performs an UPSERT-like merge using DuckDB directly on Parquet
#   (anti-join old rows on index columns + UNION BY NAME with new rows), avoiding
#   loading large Parquet files into Pandas.
# - read_dataframe: executes filters/sorts/limits via DuckDB SQL on Parquet for
#   predicate pushdown & column pruning.
# - First-write safety: when target Parquet doesn't exist, we create it via DuckDB
#   (no need to specify overwrite/append explicitly).
#
# Requirements: duckdb>=0.8 (supports UNION BY NAME) and pyarrow for Parquet IO.


class DuckDBFileStorage(LocalFileStorage):
    """Drop-in replacement that uses DuckDB for efficient reads & upserts.

    Public method names & signatures are preserved to match LocalFileStorage:
      - insert_dataframe(self, df, schema, tb_name, index, format_ext='parquet')
      - read_dataframe(self, schema, tb_name, filter_column=None, filter_keyword=None, ascending=None, limit=None, format_ext='parquet')
    """

    def __init__(self, storage_config: dict = None):
        super().__init__(storage_config)
        # DuckDB connection (can be ":memory:" or a file path like "./doc/fs/meta.duckdb")
        self.storage_config = storage_config or file_storage_config()

        self.duckdb_path = Path(project_path()) / Path(self.storage_config.get('base_path')) / Path(self.storage_config.get('duckdb'))

        self.default_format = self.storage_config.get('default_format', 'parquet')

    @staticmethod
    def _escape_string_literal(path: str) -> str:
        """
        Mimic duckdb.escape_string_literal for file paths.
        Wrap the string in single quotes and escape inner quotes.
        """
        return "'" + path.replace("'", "''") + "'"

    @contextmanager
    def _duckdb_connection(self):
        """Context manager for DuckDB connection."""
        conn = duckdb.connect(self.duckdb_path.as_posix())
        try:
            yield conn
        except Exception as e:
            logger.error(f"Error with DuckDB connection: {e}")
            raise
        finally:
            conn.close()

    # ---------------------------- Helpers ---------------------------- #
    @staticmethod
    def _q_ident(name: str) -> str:
        """Quote an identifier (column name) for SQL.
        Very conservative: double-quote and escape internal quotes.
        """
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _sql_literal(value: Any) -> str:
        """Render a Python value as a SQL literal (simple, local only)."""
        if value is None:
            return 'NULL'
        if isinstance(value, (int, float)):
            return str(value)
        # Fallback to string literal with single-quote escaping
        return "'" + str(value).replace("'", "''") + "'"

    def _build_where(self, filter_keyword: Optional[Dict[str, Dict[str, Any]]]) -> str:
        if not filter_keyword:
            return ""
        clauses = []
        for col, conds in filter_keyword.items():
            col_q = self._q_ident(col)
            for op, val in conds.items():
                if op == 'eq':
                    clauses.append(f"{col_q} = {self._sql_literal(val)}")
                elif op == 'ne':
                    clauses.append(f"{col_q} <> {self._sql_literal(val)}")
                elif op == 'gt':
                    clauses.append(f"{col_q} > {self._sql_literal(val)}")
                elif op == 'gte':
                    clauses.append(f"{col_q} >= {self._sql_literal(val)}")
                elif op == 'lt':
                    clauses.append(f"{col_q} < {self._sql_literal(val)}")
                elif op == 'lte':
                    clauses.append(f"{col_q} <= {self._sql_literal(val)}")
                elif op == 'in' and isinstance(val, list):
                    in_vals = ", ".join(self._sql_literal(v) for v in val)
                    clauses.append(f"{col_q} IN ({in_vals})")
                else:
                    # Unsupported op -> skip (keep parity with base class behavior)
                    continue
        return (" WHERE " + " AND ".join(clauses)) if clauses else ""

    # ---------------------------- Write (UPSERT w/ DuckDB) ---------------------------- #
    def insert_dataframe(self, df: pd.DataFrame, schema: str, tb_name: str,
                        index: List[str],
                        format_ext: str = 'parquet') -> None:
        """
        Efficient upsert/append to Parquet using DuckDB.
        - First write: direct COPY.
        - Subsequent writes:
            - If index: new data replaces old data with same index (anti-join + union).
            - If no index: simply append (union).
        """
        if df is None or df.empty:
            logger.warning(f"Empty DataFrame provided for {schema}.{tb_name}")
            return

        df_prepared = self._prepare_dataframe_for_storage(df.copy())
        format_to_use = format_ext or self.default_format
        self._validate_format(format_to_use, raise_missing_dependency=True)

        # Ensure directory and file path
        self._ensure_schema_dir(schema, tb_name)
        file_path = self._get_file_path(schema, tb_name, format_to_use)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with self._duckdb_connection() as conn:
            # Register new data
            try:
                conn.unregister("new_data")
            except Exception:
                pass
            conn.register("new_data", df_prepared)

            # First write: file does not exist
            if not file_path.exists():
                file_str = str(file_path.as_posix()).replace("'", "''")
                sql_first = f"COPY (SELECT * FROM new_data) TO '{file_str}' (FORMAT PARQUET)"
                conn.execute(sql_first)
                logger.info(f"Data inserted (first write) into {schema}.{tb_name}: DataFrame{df_prepared.shape}, format: {format_to_use}")
                return

            # File exists: merge
            path_escaped = str(file_path.as_posix()).replace("'", "''")
            # 获取所有列（旧+新，顺序不变）
            cols_df = conn.execute(f"SELECT * FROM read_parquet('{file_path.as_posix()}') LIMIT 1").fetchdf()
            old_cols = cols_df.columns.tolist()
            new_cols = df_prepared.columns.tolist()
            all_cols = list(dict.fromkeys(old_cols + new_cols))

            def build_select(table_alias, available_cols):
                return ", ".join(
                    f"{table_alias}.{self._q_ident(c)}" if c in available_cols else f"NULL AS {self._q_ident(c)}"
                    for c in all_cols
                )

            if index and len(index) > 0:
                # 去重：新数据覆盖旧数据（按index去重）
                idx_conds = " AND ".join([f"t.{self._q_ident(c)} = n.{self._q_ident(c)}" for c in index])
                merged_select = f"""
                    WITH old_filtered AS (
                        SELECT {build_select('t', old_cols)}
                        FROM read_parquet('{path_escaped}') AS t
                        WHERE NOT EXISTS (
                            SELECT 1 FROM new_data AS n WHERE {idx_conds}
                        )
                    )
                    SELECT * FROM old_filtered
                    UNION BY NAME
                    SELECT {build_select('n', new_cols)} FROM new_data AS n
                """
            else:
                # 直接追加
                merged_select = f"""
                    SELECT {build_select('t', old_cols)}
                    FROM read_parquet('{path_escaped}') AS t
                    UNION BY NAME
                    SELECT {build_select('n', new_cols)} FROM new_data AS n
                """

            # 写入临时文件，再原子替换
            tmp_path = file_path.with_suffix('.tmp.parquet')
            tmp_str = str(tmp_path.as_posix()).replace("'", "''")
            sql_merge = f"COPY ({merged_select}) TO '{tmp_str}' (FORMAT PARQUET)"
            conn.execute(sql_merge)
            try:
                tmp_path.replace(file_path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        logger.info(f"Data inserted (merged) into {schema}.{tb_name}: DataFrame{df_prepared.shape}, format: {format_to_use}")


    # ---------------------------- Read (via DuckDB) ---------------------------- #
    def read_dataframe(self, schema: str, tb_name: str,
                       filter_column: Optional[List[str]] = None,
                       filter_keyword: Optional[Dict[str, Dict[str, Any]]] = None,
                       ascending: Optional[List[tuple]] = None,
                       limit: Optional[int] = None,
                       format_ext: str = 'parquet') -> pd.DataFrame:
        """Read with predicate pushdown & column pruning using DuckDB over Parquet."""
        requested_format = format_ext or self.default_format
        self._validate_format(requested_format, raise_missing_dependency=True)

        existing_format = self._detect_existing_format(schema, tb_name)
        if not existing_format:
            logger.warning(f"No file found for {schema}.{tb_name}")
            return pd.DataFrame()
        if existing_format != requested_format:
            raise ValueError(
                f"Existing table {schema}.{tb_name} format {existing_format} does not match requested {requested_format}"
            )

        file_path = self._get_file_path(schema, tb_name, existing_format)
        if not file_path.exists():
            logger.warning(f"File path not found for read: {file_path}")
            return pd.DataFrame()

        # Build base SQL
        select_cols = "*"
        if filter_column:
            safe_cols = [self._q_ident(c) for c in filter_column]
            select_cols = ", ".join(safe_cols) if safe_cols else "*"

        where_sql = self._build_where(filter_keyword)

        order_sql = ""
        if ascending:
            parts = []
            for col, asc in ascending:
                parts.append(f"{self._q_ident(col)} {'ASC' if asc else 'DESC'}")
            if parts:
                order_sql = " ORDER BY " + ", ".join(parts)

        limit_sql = f" LIMIT {int(limit)}" if limit and limit > 0 else ""

        sql = f"SELECT {select_cols} FROM read_parquet('{file_path.as_posix()}'){where_sql}{order_sql}{limit_sql}"
        try:
            with self._duckdb_connection() as conn:
                df = conn.execute(sql).fetchdf()
        except Exception as e:
            logger.error(f"DuckDB read failed for {schema}.{tb_name}: {e}")
            return pd.DataFrame()

        # Keep parity with base class post-processing
        df = self._normalize_datetime_columns(df)
        return df.reset_index(drop=True)
