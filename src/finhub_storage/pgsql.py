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
        self.__cursor__ = None
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
        self.__cursor__ = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self.config = cfg
    def __ensure_connection__(self):
        self.__connect__()

    def __ensure_schema_exists__(self, schema: str):
        self.__ensure_connection__()
        with self.__get_cursor__() as cursor:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
            self.conn.commit()

    @contextmanager
    def __get_cursor__(self):
        """内部使用的游标上下文管理器：确保已连接后返回游标"""
        self.__ensure_connection__()
        yield self.__cursor__

    def __execute_sql__(self, sql: str):
        try:
            with self.__get_cursor__() as cursor:
                cursor.execute(sql)
                self.conn.commit()
                # 只有 SELECT 语句才 fetchall
                if sql.strip().lower().startswith("select"):
                    return cursor.fetchall()
                else:
                    return None
        except Exception as e:
            # 事务失败后必须 rollback，否则后续 SQL 都会被忽略
            if self.conn and not self.conn.closed:
                try:
                    self.conn.rollback()
                except Exception as re:
                    logger.error(f"Rollback failed: {re}")
            raise e

    def read_by_sql(self, sql_str: str):
        """执行SQL查询并返回结果"""
        try:
            with self.__get_cursor__() as cursor:
                cursor.execute(sql_str)
                return cursor.fetchall()
        except Exception as e:
            raise Exception(f"Query execution failed: {e}")

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

    @staticmethod
    def __map_table_types__(df: pd.DataFrame):
        type_dic = df.dtypes.to_dict()
        type_dic_for_pg = {}
        for k, v in type_dic.items():
            col_lower = str(k).lower()
            vt = str(v).lower()
            # 优先按列名规则：包含 'date' 或 'time' 的列统一为 BIGINT（epoch ns）
            if ('date' in col_lower) or ('time' in col_lower):
                type_dic_for_pg[k] = 'TIMESTAMP'
            elif 'int' in vt:
                type_dic_for_pg[k] = 'INTEGER'
            elif 'float' in vt:
                type_dic_for_pg[k] = 'FLOAT'
            else:
                type_dic_for_pg[k] = 'TEXT'
        return type_dic_for_pg

    @staticmethod
    def __alter_df_type_for_sql__(df: pd.DataFrame):
        for k, v in df.dtypes.to_dict().items():
            col_lower = str(k).lower()
            vt = str(v).lower()
            if ('date' in col_lower) or ('time' in col_lower):
                # 列名包含 date/time：统一转为 Python naive datetime（不含时区），以 TIMESTAMP 存储
                s = pd.to_datetime(df[k], errors='coerce')
                try:
                    # 若为 tz-aware，去除时区信息但不进行时区换算（保留表面时间）
                    if isinstance(s.dtype, pd.DatetimeTZDtype):
                        s = s.dt.tz_localize(None)
                except Exception:
                    # 回退逐元素处理
                    s = s.apply(
                        lambda x: x.replace(tzinfo=None) if (pd.notna(x) and getattr(x, 'tzinfo', None)) else x
                    )
                # 转 Python datetime；NaT -> None
                df[k] = s.apply(lambda x: x.to_pydatetime() if pd.notna(x) else None)
            else:
                if 'int' in vt:
                    df[k] = df[k].astype(int)
                elif 'float' in vt:
                    df[k] = df[k].astype(float)
                else:
                    df[k] = df[k].astype('str').fillna('')
        return df

    @staticmethod
    def __forge_primary_id__(df: pd.DataFrame, index):
        # 检查联合主键字段是否有缺失
        if df[index].isnull().values.any():
            raise ValueError("Index dataframe contains NaN values, cannot forge primary key.")
        # 不再拼接 id，直接返回原 df
        return df

    def __check_table__(self, schema: str, tb_name: str, create_table_on_error: bool = True, df: pd.DataFrame = None, index: list = None):
        self.__ensure_schema_exists__(schema)
        try:
            self.__execute_sql__(f'SELECT 1 FROM "{schema}"."{tb_name}" LIMIT 1;')
            return True
        except Exception as e:
            if create_table_on_error and (('does not exist' in str(e)) or ('不存在' in str(e))):
                self.create_table(df, schema, tb_name, index)
                return True
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
        with self.__get_cursor__() as cursor:
            psycopg2.extras.execute_values(cursor, sql, records, page_size=500)
        self.conn.commit()

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

    @staticmethod
    def __operand_map_for_sql__(input_) -> str:
        return {
            "gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "eq": "=", "ne": "<>"
        }.get(input_)

    def __make_filters__(self, filter_: Union[dict, None] = None):
        sql_fw = ""
        if filter_:
            for k, v in filter_.items():
                col_lower = str(k).lower()
                for op, val in v.items():
                    if op == 'in' and isinstance(val, list):
                        parts = []
                        for i in val:
                            if ('date' in col_lower or 'time' in col_lower) and isinstance(i, (pd.Timestamp, datetime)):
                                s = i.strftime('%Y-%m-%d %H:%M:%S')
                                parts.append(f"'{s}'")
                            elif isinstance(i, (int, float)) and not isinstance(i, bool):
                                parts.append(str(i))
                            else:
                                parts.append(f"'{i}'")
                        csv = ', '.join(parts)
                        sql_fw += f'"{k}" IN ({csv})'
                    else:
                        if ('date' in col_lower or 'time' in col_lower) and isinstance(val, (pd.Timestamp, datetime)):
                            s = val.strftime('%Y-%m-%d %H:%M:%S')
                            val_str = f"'{s}'"
                        elif isinstance(val, (int, float)) and not isinstance(val, bool):
                            val_str = str(val)
                        else:
                            val_str = f"'{val}'"
                        sql_fw += f'"{k}" {self.__operand_map_for_sql__(op)} {val_str}'
                    sql_fw += ' AND '
        return sql_fw.rstrip(' AND ') if sql_fw else ""

    def read_dataframe(self, schema: str, tb_name: str, filter_column: list = None,
                       filter_keyword: dict = None, ascending: list = None,
                       limit: int = None):
        if not self.__check_table__(schema, tb_name, False):
            logger.warning(f"No table found: {schema}.{tb_name}")
            return pd.DataFrame()

        col_param = ', '.join([f'"{i}"' for i in filter_column]) if filter_column else "*"
        where_clause = self.__make_filters__(filter_keyword)
        sql = f'SELECT {col_param} FROM "{schema}"."{tb_name}"'
        if where_clause:
            sql += f" WHERE {where_clause}"
        if ascending:
            col, asc = ascending[0]
            sql += f' ORDER BY "{col}" {"ASC" if asc else "DESC"}'
        if limit:
            sql += f' LIMIT {limit}'
        sql += ";"

        rows = self.__execute_sql__(sql)
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
