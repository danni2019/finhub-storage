import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime
from typing import Union
from contextlib import contextmanager
from infra.tool.config import postgres_config
from infra.tool.log import log


logger = log(__file__, 'database')

"""
finHub database ORM -- PostgreSQL
Datetime列规范：
- 列名包含 'date' 或 'time' 的列：写入时统一转换为 Python naive datetime（不含时区），以 TIMESTAMP 存储；保留微秒；无效值写 NULL。
- 读取时自动将此类列转为 pandas datetime64[ns]；时区语义留给业务层处理。
- 对外 API 与参数不变。
"""

class PostgreSQL:
    def __init__(self, timeout: int = 15):
        """
        Initialize PostgreSQL connection manager.
        Args:
            timeout: Connection timeout in seconds
        """
        self.config = postgres_config()
        # 配置中应提供 database（数据库名）；若缺失则默认使用 postgres

        self.timeout = timeout
        self.conn = None
        self.__connect__()

    @staticmethod
    def __clear_config__(conf: dict):
        allowed_keys = {'host', 'port', 'user', 'password', 'database', 'dbname', 'sslmode', 'connect_timeout'}
        clean_cfg = {k: v for k, v in conf.items() if k in allowed_keys}
        # 兼容：优先使用 'database'，转为 psycopg2 接受的 'dbname'
        if 'database' in clean_cfg:
            clean_cfg['dbname'] = clean_cfg.pop('database')
        return clean_cfg

    def __connect__(self):
        cfg = self.__clear_config__(dict(self.config))
        if not cfg.get('dbname'):
            cfg['dbname'] = 'postgres'
        # 若未连接或连接关闭则连接；若目标库名变化则重连
        need_reconnect = (not self.conn) or self.conn.closed or (cfg.get('dbname') != self.config.get('dbname'))
        if not need_reconnect:
            return
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
        except Exception:
            pass
        self.conn = psycopg2.connect(**cfg, connect_timeout=self.timeout)
        self.config = cfg

    def __ensure_connection__(self):
        self.__connect__()

    # 事务上下文
    @contextmanager
    def transaction(self):
        self.__ensure_connection__()
        try:
            yield
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def __ensure_schema_exists__(self, schema: str):
        self.__ensure_connection__()
        with self.transaction():
            with self.__get_cursor__() as cursor:
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')

    @contextmanager
    def __get_cursor__(self):
        self.__ensure_connection__()
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def __execute_sql__(self, sql: str, params: tuple | list | None = None):
        try:
            with self.transaction():
                with self.__get_cursor__() as cursor:
                    cursor.execute(sql, params)
                    if cursor.description:  # 有结果集
                        return cursor.fetchall()
                    return None
        except Exception as e:
            if self.conn and not self.conn.closed:
                try:
                    self.conn.rollback()
                except Exception as re:
                    logger.error(f"Rollback failed: {re}")
            raise

    def read_by_sql(self, sql_str: str):
        """执行SQL查询并返回结果"""
        return self.__execute_sql__(sql_str)

    def is_timescaledb_enabled(self):
        """检测是否启用 TimescaleDB 扩展"""
        sql = "SELECT extname FROM pg_extension WHERE extname='timescaledb';"
        result = self.__execute_sql__(sql)
        return bool(result)

    def create_hypertable(self, schema, tb_name, time_col):
        if not self.is_timescaledb_enabled():
            return
        if not time_col:
            return
        # '"schema"."table"' 作为 text 再转 regclass，time_col 也加双引号
        rel = f'"{schema}"."{tb_name}"'
        sql = "SELECT create_hypertable(%s::regclass, %s, if_not_exists => TRUE);"
        try:
            self.__execute_sql__(sql, params=[rel, time_col])
        except Exception as e:
            logger.warning(f"create_hypertable failed: {e}")


    def create_table(self, df: pd.DataFrame, schema: str, tb_name: str, index: list):
        if df is None or df.empty:
            raise ValueError("Cannot create table with empty DataFrame.")
        for col in index:
            if col not in df.columns:
                raise ValueError(f"Primary key column '{col}' is missing in DataFrame.")
        self.__ensure_schema_exists__(schema)
        datatype_dic = self.__map_table_types__(df)
        columns = [f'"{k}" {v}' for k, v in datatype_dic.items()]
        pk_fields = ', '.join([f'"{col}"' for col in index])
        columns_str = ", ".join(columns + [f'PRIMARY KEY ({pk_fields})'])
        create_table_sql = f'CREATE TABLE IF NOT EXISTS "{schema}"."{tb_name}" ({columns_str});'
        self.__execute_sql__(create_table_sql)
        """
        此处会自动检测index并为时序数据创建hypertable。
        注意！这里隐藏了一个业务逻辑：只有含datetime或timestamp的列才会被用作创建hypertable的时间轴。
        其他类型的列将被忽略，即使是含有date或time，也不会创建为hypertable，避免稀疏日期数据的复杂处理。
        """
        time_cols = [col for col in index if 'datetime' == col.lower() or 'timestamp' == col.lower()]
        if time_cols:
            self.create_hypertable(schema, tb_name, time_cols[0])
            self.create_time_indexes(schema, tb_name, time_cols)

    def create_time_indexes(self, schema, tb_name, time_cols, dim_cols: list[str] | None = None):
        for col in time_cols:
            self.__execute_sql__(f'CREATE INDEX IF NOT EXISTS "{tb_name}_{col}_idx" ON "{schema}"."{tb_name}" ("{col}");')
            if dim_cols:
                for d in dim_cols:
                    idx = f'{tb_name}_{d}_{col}_idx'
                    self.__execute_sql__(f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{schema}"."{tb_name}" ("{d}", "{col}" DESC);')

    @staticmethod
    def __map_table_types__(df: pd.DataFrame):
        type_dic_for_pg = {}
        for k, v in df.dtypes.to_dict().items():
            col_lower = str(k).lower()
            vt = str(v).lower()
            if ('date' in col_lower) or ('time' in col_lower):
                type_dic_for_pg[k] = 'TIMESTAMP'
            elif vt == 'bool' or vt == 'boolean':
                type_dic_for_pg[k] = 'BOOLEAN'
            elif 'int' in vt:
                type_dic_for_pg[k] = 'BIGINT' if 'int64' in vt else 'INTEGER'
            elif 'float' in vt:
                type_dic_for_pg[k] = 'DOUBLE PRECISION'
            else:
                type_dic_for_pg[k] = 'TEXT'
        return type_dic_for_pg


    @staticmethod
    def __alter_df_type_for_sql__(df: pd.DataFrame):
        for k, v in df.dtypes.to_dict().items():
            col_lower = str(k).lower()
            vt = str(v).lower()

            if ('date' in col_lower) or ('time' in col_lower):
                s = pd.to_datetime(df[k], errors='coerce')
                try:
                    if isinstance(s.dtype, pd.DatetimeTZDtype):
                        s = s.dt.tz_localize(None)
                except Exception:
                    s = s.apply(lambda x: x.replace(tzinfo=None) if (pd.notna(x) and getattr(x, 'tzinfo', None)) else x)
                df[k] = s.apply(lambda x: x.to_pydatetime() if pd.notna(x) else None).astype(object)

            elif 'int' in vt:
                s = pd.to_numeric(df[k], errors='coerce')
                s = s.astype('Int64')
                df[k] = s.where(s.notna(), None).astype(object)

            elif 'float' in vt:
                s = pd.to_numeric(df[k], errors='coerce')
                df[k] = s.where(pd.notna(s), None).astype(object)

            elif vt == 'bool':
                s = df[k].astype('boolean')
                df[k] = s.where(s.notna(), None).astype(object)

            else:
                # 文本列也尽量用 None 表示 NULL，而不是空字符串
                s = df[k].astype(object)
                df[k] = s.where(pd.notna(s), None)
        return df


    @staticmethod
    def __forge_primary_id__(df: pd.DataFrame, index):
        # 检查联合主键字段是否有缺失
        if df[index].isnull().values.any():
            raise ValueError("Index dataframe contains NaN values, cannot forge primary key.")
        # 不再拼接 id，直接返回原 df
        return df

    def __check_table__(self, schema: str, tb_name: str, create_table_on_error: bool = True,
                        df: pd.DataFrame = None, index: list = None):
        self.__ensure_schema_exists__(schema)
        try:
            self.__execute_sql__(f'SELECT 1 FROM "{schema}"."{tb_name}" LIMIT 1;')
            return True
        except psycopg2.errors.UndefinedTable:
            if create_table_on_error and df is not None and index is not None:
                self.create_table(df, schema, tb_name, index)
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking table {schema}.{tb_name}: {e}")
            return False

    def __write_df_into_table__(self, df: pd.DataFrame, schema: str, tb_name: str, index: list):
        columns = ','.join([f'"{c}"' for c in df.columns.tolist()])
        # 联合主键字段
        pk_fields = ', '.join([f'"{col}"' for col in index])
        # ON CONFLICT 语句用联合主键
        sql = (
            f'INSERT INTO "{schema}"."{tb_name}" ({columns}) VALUES %s '
            f'ON CONFLICT ({pk_fields}) DO UPDATE SET '
            + ', '.join([f'"{c}" = EXCLUDED."{c}"' for c in df.columns if c not in index])
        )
        records = list(df.itertuples(index=False, name=None))
        with self.transaction():
            with self.__get_cursor__() as cursor:
                psycopg2.extras.execute_values(cursor, sql, records, page_size=500)

    def __ensure_columns_exist__(self, df: pd.DataFrame, schema: str, tb_name: str):
        existing_cols = self.get_col_names(schema, tb_name)
        missing_cols = [col for col in df.columns.tolist() if col not in existing_cols]
        if not missing_cols:
            return
        type_mapping = self.__map_table_types__(df)
        for col in missing_cols:
            col_type = type_mapping.get(col, 'TEXT')
            alter_sql = f'ALTER TABLE "{schema}"."{tb_name}" ADD COLUMN "{col}" {col_type};'
            self.__execute_sql__(alter_sql)

    def insert_dataframe(self, df: pd.DataFrame, schema: str, tb_name: str, index: list, auto_extend_col: bool = False):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return
        self.__alter_df_type_for_sql__(df)
        df = self.__forge_primary_id__(df, index)
        self.__check_table__(schema, tb_name, True, df, index)
        if auto_extend_col:
            self.__ensure_columns_exist__(df, schema, tb_name)
        self.__write_df_into_table__(df, schema, tb_name, index)
        logger.info(f"Data inserted into {schema}.{tb_name}: Dataframe{df.shape}")

    def get_col_names(self, schema: str, tb_name: str):
        sql = (
            f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = '{schema}' AND table_name = '{tb_name}'
            ORDER BY ordinal_position;
            """
        )
        rows = self.__execute_sql__(sql)
        return [r['column_name'] for r in rows] if rows else []

    def __operand_map_for_sql__(self, input_) -> str:
        op = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "eq": "=", "ne": "<>"}.get(input_)
        if not op:
            raise ValueError(f"Unsupported operator: {input_}")
        return op

    def __make_filters__(self, filter_: Union[dict, None] = None):
        if not filter_:
            return "", []
        clauses, params = [], []
        for k, cond in filter_.items():
            for op, val in cond.items():
                if op == "in":
                    if not isinstance(val, (list, tuple)) or len(val) == 0:
                        # 空 IN () 是非法的；直接构造 false 条件
                        clauses.append("1=0")
                        continue
                    placeholders = ", ".join(["%s"] * len(val))
                    clauses.append(f'"{k}" IN ({placeholders})')
                    params.extend(val)
                else:
                    sqlop = self.__operand_map_for_sql__(op)
                    clauses.append(f'"{k}" {sqlop} %s')
                    params.append(val)
        return " AND ".join(clauses), params


    def read_dataframe(self, schema: str, tb_name: str, filter_column: list = None,
                    filter_keyword: dict = None, ascending: list = None,
                    limit: int = None):
        if not self.__check_table__(schema, tb_name, False):
            logger.warning(f"No table found: {schema}.{tb_name}")
            return pd.DataFrame()

        col_param = ', '.join([f'"{i}"' for i in filter_column]) if filter_column else "*"
        where_sql, params = self.__make_filters__(filter_keyword)

        sql = f'SELECT {col_param} FROM "{schema}"."{tb_name}"'
        if where_sql:
            sql += f" WHERE {where_sql}"
        if ascending:
            order_sql = ', '.join([f'"{c}" {"ASC" if a else "DESC"}' for c, a in ascending])
            sql += f' ORDER BY {order_sql}'
        if limit:
            sql += f' LIMIT {limit}'
        sql += ";"

        rows = self.__execute_sql__(sql, params)
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            for col in df.columns:
                col_lower = str(col).lower()
                if ('date' in col_lower) or ('time' in col_lower):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Failed to convert column {col} to datetime: {e}")
        return df


    def del_row(self, schema: str, tb_name: str, filter_sql: str):
        sql = f'DELETE FROM "{schema}"."{tb_name}" WHERE {filter_sql};'
        self.__execute_sql__(sql)

    def drop_table(self, schema: str, tb_name: str):
        sql = f'DROP TABLE IF EXISTS "{schema}"."{tb_name}";'
        self.__execute_sql__(sql)

    def get_table_name(self, schema: str):
        """获取指定 schema 下的所有表名"""
        sql = f"SELECT tablename FROM pg_tables WHERE schemaname='{schema}' ORDER BY tablename;"
        rows = self.__execute_sql__(sql)
        return [i['tablename'] for i in rows] if rows else []

    def exists(self, schema: str, tb_name: str, filter: dict) -> bool:
        """检查是否存在满足条件的记录"""
        if not filter:
            return False
        
        where_clause = ' AND '.join([f'"{k}" = %s' for k in filter.keys()])
        sql = f"""SELECT 1 FROM "{schema}"."{tb_name}" WHERE {where_clause} LIMIT 1;"""
        try:
            with self.__get_cursor__() as cursor:
                cursor.execute(sql, list(filter.values()))
                return bool(cursor.fetchone())
        except Exception as e:
            logger.error(f"Exists check failed for {schema}.{tb_name}: {e}")
            return False
        
    def _safe_agg(self, agg_func: str):
        agg = agg_func.lower()
        if agg not in {'avg','sum','min','max','count','stddev','percentile_cont'}:
            raise ValueError(f"Unsupported agg_func: {agg_func}")
        return agg

    def time_bucket_query(self, schema, tb_name, time_col, bucket, agg_col, agg_func='avg', filter_keyword=None):
        if not self.is_timescaledb_enabled():
            raise RuntimeError("TimescaleDB not enabled: time_bucket requires TimescaleDB.")
        agg_func = self._safe_agg(agg_func)
        where_sql, params = self.__make_filters__(filter_keyword)
        sql = (
            f'SELECT time_bucket(%s::interval, "{time_col}") AS bucket, {agg_func}("{agg_col}") AS value '
            f'FROM "{schema}"."{tb_name}"'
        )
        if where_sql:
            sql += f' WHERE {where_sql}'
        sql += ' GROUP BY bucket ORDER BY bucket;'
        rows = self.__execute_sql__(sql, params=[bucket] + params)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def gapfill_query(self, schema, tb_name, time_col, bucket, agg_col, agg_func='avg', filter_keyword=None):
        if not self.is_timescaledb_enabled():
            raise RuntimeError("TimescaleDB not enabled: time_bucket_gapfill requires TimescaleDB.")
        agg_func = self._safe_agg(agg_func)
        where_sql, params = self.__make_filters__(filter_keyword)
        sql = (
            f'SELECT time_bucket_gapfill(%s::interval, "{time_col}") AS bucket, {agg_func}("{agg_col}") AS value '
            f'FROM "{schema}"."{tb_name}"'
        )
        if where_sql:
            sql += f' WHERE {where_sql}'
        sql += ' GROUP BY bucket ORDER BY bucket;'
        rows = self.__execute_sql__(sql, params=[bucket] + params)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    def enable_compression(self, schema, tb_name):
        try:
            self.__execute_sql__(f'ALTER TABLE "{schema}"."{tb_name}" SET (timescaledb.compress);')
        except Exception as e:
            logger.warning(f"Enable compression failed: {e}")

    def add_compression_policy(self, schema, tb_name, time_col=None, compress_after='7 days'):
        if not self.is_timescaledb_enabled():
            logger.warning("TimescaleDB not enabled: skip compression policy.")
            return
        self.enable_compression(schema, tb_name)
        sql = "SELECT add_compression_policy(%s::regclass, %s::interval);"
        rel = f'"{schema}"."{tb_name}"'
        try:
            self.__execute_sql__(sql, params=[rel, compress_after])
        except Exception as e:
            logger.warning(f"Add compression policy failed: {e}")

    def add_retention_policy(self, schema, tb_name, drop_after='30 days'):
        if not self.is_timescaledb_enabled():
            logger.warning("TimescaleDB not enabled: skip retention policy.")
            return
        sql = "SELECT add_retention_policy(%s::regclass, %s::interval);"
        rel = f'"{schema}"."{tb_name}"'
        try:
            self.__execute_sql__(sql, params=[rel, drop_after])
        except Exception as e:
            logger.warning(f"Add retention policy failed: {e}")

