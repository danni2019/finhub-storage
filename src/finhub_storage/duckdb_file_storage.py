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
        """Upsert-like insert using DuckDB to avoid loading large Parquet into Pandas.

        Semantics:
          - If file doesn't exist: create it from df (first write path).
          - If file exists & index provided: remove existing rows with matching index
            from old Parquet, then UNION BY NAME with new rows (dedup on index).
          - If file exists & no index provided: append-all semantics via UNION BY NAME.

        Notes:
          - Parquet is immutable; we rewrite the file by streaming through DuckDB.
          - DataFrame Pandas index is *not* stored (same as your original code).
        """
        if df is None or df.empty:
            logger.warning(f"Empty DataFrame provided for {schema}.{tb_name}")
            return

        # Prepare & validate
        df_prepared = self._prepare_dataframe_for_storage(df.copy())
        format_to_use = format_ext or self.default_format
        self._validate_format(format_to_use, raise_missing_dependency=True)

        # Ensure directory exists and resolve file path
        self._ensure_schema_dir(schema, tb_name)
        file_path = self._get_file_path(schema, tb_name, format_to_use)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Register incoming frame as a DuckDB temp view
        with self._duckdb_connection() as conn:
            conn.unregister("new_data") if "new_data" in conn.get_table_names() else None
            conn.register("new_data", df_prepared)

            # First write: create file directly via DuckDB COPY
            if not file_path.exists():
                sql_first = "COPY (SELECT * FROM new_data) TO ? (FORMAT 'parquet')"
                conn.execute(sql_first, [str(file_path.as_posix())])
                logger.info(f"Data inserted (first write) into {schema}.{tb_name}: DataFrame{df_prepared.shape}, format: {format_to_use}")
                return

            # File exists: build merged SELECT using DuckDB over Parquet
            path_str = file_path.as_posix()

            if index and len(index) > 0:
                # Anti-join on index columns to drop conflicting old rows
                idx_conds = " AND ".join([f"t.{self._q_ident(c)} = n.{self._q_ident(c)}" for c in index])
                merged_select = f"""
                    WITH old_filtered AS (
                        SELECT *
                        FROM read_parquet('{path_str}') AS t
                        WHERE NOT EXISTS (
                            SELECT 1 FROM new_data AS n WHERE {idx_conds}
                        )
                    )
                    SELECT * FROM old_filtered
                    UNION BY NAME ALL
                    SELECT * FROM new_data
                """
            else:
                # No index: pure append (no dedup)
                merged_select = f"""
                    SELECT * FROM read_parquet('{path_str}')
                    UNION BY NAME ALL
                    SELECT * FROM new_data
                """

            # Write to a temp parquet then atomically replace
            tmp_path = file_path.with_suffix('.tmp.parquet')
            conn.execute(f"COPY ({merged_select}) TO ? (FORMAT 'parquet')", [str(tmp_path.as_posix())])
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
