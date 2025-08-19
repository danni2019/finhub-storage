import os
import json
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from datetime import datetime
import warnings

from .file_storage_base import FileStorageBase

# Try to use the project's logging system, fall back to simple logger
try:
    from infra.tool.log import log
    logger = log(__file__, 'data')
except (ImportError, FileNotFoundError):
    # Simple logger fallback for when config system is not available
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    logger = SimpleLogger()

# Try to import configuration system
try:
    from infra.tool.config import file_storage_config
    CONFIG_AVAILABLE = True
except (ImportError, FileNotFoundError):
    CONFIG_AVAILABLE = False
    logger.warning("Configuration system not available, using default file storage settings")


class LocalFileStorage(FileStorageBase):
    """
    Local file storage implementation
    
    This class implements file storage on the local filesystem, supporting multiple
    file formats (CSV, JSON, Excel, Parquet) with automatic format detection.
    
    File organization:
    - base_path/schema/tb_name.{format}
    - Schemas are implemented as directories
    - Tables are implemented as files
    """
    
    SUPPORTED_FORMATS = ['csv', 'json', 'excel']
    DEFAULT_FORMAT = 'parquet'  # Parquet for best compatibility with ArcticDB
    
    def __init_class_formats__(self):
        """Initialize supported formats based on available libraries

        Parquet support is only enabled when pyarrow is available. No silent
        fallback is performed; if the user/config requests parquet but the
        dependency is missing, an exception will be raised later during
        validation.
        """
        try:  # noqa: F401
            import pyarrow  # type: ignore
            if 'parquet' not in self.SUPPORTED_FORMATS:
                self.SUPPORTED_FORMATS.append('parquet')
        except ImportError:
            # Do not modify defaults here; validation later will raise if needed
            logger.info("pyarrow not available; parquet format will be unsupported")
    
    def __init__(self, storage_config: dict = None):
        """
        Initialize local file storage
        
        Args:
            storage_config: Configuration dictionary containing:
                - base_path: Base directory for storage (default: from config system or ./doc/fs)
                - default_format: Default file format (default: from config system or parquet)
                - create_dirs: Whether to create directories automatically (default: from config system or True)
                
        Note: If storage_config is provided, it takes precedence over the config system.
              This maintains backward compatibility while enabling config system integration.
        """
        # Load configuration from config system if available and no explicit config provided
        if storage_config is None and CONFIG_AVAILABLE:
            try:
                storage_config = file_storage_config()
                logger.info("File storage configuration loaded from config system")
            except Exception as e:
                logger.warning(f"Failed to load config from system: {e}, using defaults")
                storage_config = {}
        
        # Use provided config or default fallback
        self.config = storage_config or {}
        base_path_config = self.config.get('base_path', './doc/fs')
        
        # 确保使用绝对路径
        if Path(base_path_config).is_absolute():
            self.base_path = Path(base_path_config)
        else:
            # 对于相对路径，基于项目根目录进行解析
            try:
                from infra.tool.config import project_path
                self.base_path = (Path(project_path()) / base_path_config).resolve()
            except (ImportError, Exception):
                # 如果无法获取项目路径，使用当前工作目录
                self.base_path = Path(base_path_config).resolve()
        
        self.default_format = self.config.get('default_format', self.DEFAULT_FORMAT)
        self.create_dirs = self.config.get('create_dirs', True)
        
        # Initialize supported formats based on available libraries
        self.__init_class_formats__()

        # Validate selected default format (must be supported & dependencies present)
        self._validate_format(self.default_format, raise_missing_dependency=True)
        
        # Ensure base path exists
        if self.create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"LocalFileStorage initialized with base_path: {self.base_path}, default_format: {self.default_format}")

    # ------------------------------------------------------------------
    # Internal helpers for format validation / path derivation
    # ------------------------------------------------------------------

    def _validate_format(self, format_ext: str, raise_missing_dependency: bool = False) -> bool:
        """Validate requested format.

        Args:
            format_ext: Requested file format.
            raise_missing_dependency: If True, raise when format unsupported
                instead of returning False.

        Returns:
            bool indicating support when raise_missing_dependency is False.
        """
        # Refresh supported formats in case environment changed dynamically
        if format_ext not in self.SUPPORTED_FORMATS:
            if raise_missing_dependency:
                raise ValueError(f"Unsupported file format requested: {format_ext}. Supported: {self.SUPPORTED_FORMATS}")
            return False

        if format_ext == 'parquet':
            try:  # noqa: F401
                import pyarrow  # type: ignore
            except ImportError:
                if raise_missing_dependency:
                    raise ImportError("pyarrow is required for parquet format but is not installed")
                return False
        return True
    
    def _parse_table_name(self, tb_name: str) -> tuple:
        """
        Parse table name to extract status and country according to naming convention
        
        According to data/readme.md, all table names follow the pattern:
        {status}_{country}_{other_fields}
        where status is first field (raw/processed) and country is second field (cn/us/eu)
        
        Args:
            tb_name: Table name following the naming convention
            
        Returns:
            Tuple of (status, country, remaining_name)
            If parsing fails, returns (None, None, tb_name)
        """
        parts = tb_name.split('_')
        if len(parts) >= 3:
            status = parts[0]
            country = parts[1]
            
            # Validate status and country according to naming convention
            valid_status = ['raw', 'processed']
            valid_country = ['cn', 'us', 'eu']
            
            if status in valid_status and country in valid_country:
                remaining_name = '_'.join(parts[2:])
                return status, country, remaining_name
        
        # Fallback for non-conforming table names
        logger.warning(f"Table name '{tb_name}' does not follow naming convention, using fallback")
        return None, None, tb_name

    def _get_file_path(self, schema: str, tb_name: str, format_ext: str = None) -> Path:
        """
        Get the full file path for a table with hierarchical directory structure
        
        Creates hierarchical path: base_path/schema/status/country/remaining_name.format
        Falls back to base_path/schema/tb_name.format for non-conforming names
        
        Args:
            schema: Schema name (included in hierarchical path structure)
            tb_name: Table name (filename base)
            format_ext: File format extension (default: self.default_format)
            
        Returns:
            Path object for the file
        """
        if format_ext is None:
            format_ext = self.default_format
        
        status, country, remaining_name = self._parse_table_name(tb_name)
        
        if status and country:
            # Use hierarchical structure: base_path/schema/status/country/remaining_name.format
            dir_path = self.base_path / schema / status / country
            file_path = dir_path / f"{remaining_name}.{format_ext}"
        else:
            # Fallback to original structure for non-conforming names
            schema_path = self.base_path / schema
            file_path = schema_path / f"{tb_name}.{format_ext}"
        
        return file_path
    
    def _ensure_schema_dir(self, schema: str, tb_name: str = None) -> None:
        """
        Ensure directory exists for the given schema and table name
        
        If tb_name follows naming convention, creates schema/status/country directories
        Otherwise creates traditional schema directory
        
        Args:
            schema: Schema name (included in hierarchical path structure)
            tb_name: Table name (optional, used to determine directory structure)
        """
        if tb_name:
            status, country, _ = self._parse_table_name(tb_name)
            if status and country:
                # Create hierarchical directories: base_path/schema/status/country/
                dir_path = self.base_path / schema / status / country
            else:
                # Fallback to schema directory for non-conforming names
                dir_path = self.base_path / schema
        else:
            # Traditional schema directory when no table name provided
            dir_path = self.base_path / schema
        
        if self.create_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _detect_existing_format(self, schema: str, tb_name: str) -> Optional[str]:
        """
        Detect if a file exists in any supported format
        
        Args:
            schema: Schema name
            tb_name: Table name
            
        Returns:
            Format of existing file or None if not found
        """
        for fmt in self.SUPPORTED_FORMATS:
            file_path = self._get_file_path(schema, tb_name, fmt)
            if file_path.exists():
                return fmt
        return None
    
    def _save_dataframe(self, df: pd.DataFrame, file_path: Path, format_ext: str) -> None:
        """
        Save DataFrame to file in specified format
        
        Args:
            df: DataFrame to save
            file_path: Target file path
            format_ext: File format extension
        """
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_ext == 'csv':
            df.to_csv(file_path, index=False, encoding='utf-8')
        elif format_ext == 'json':
            # Convert datetime columns to ISO format for JSON serialization
            df_json = df.copy()
            for col in df_json.columns:
                if pd.api.types.is_datetime64_any_dtype(df_json[col]):
                    df_json[col] = df_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_json.to_json(file_path, orient='records', indent=2, force_ascii=False)
        elif format_ext == 'excel':
            df.to_excel(file_path, index=False, engine='openpyxl')
        elif format_ext == 'parquet':
            # No fallback allowed per new requirements
            try:
                df.to_parquet(file_path, index=False, engine='pyarrow')
            except ImportError as e:
                raise ImportError("pyarrow is required to write parquet files; install pyarrow or choose another format") from e
        else:
            raise ValueError(f"Unsupported format: {format_ext}")
    
    def _load_dataframe(self, file_path: Path, format_ext: str) -> pd.DataFrame:
        """
        Load DataFrame from file in specified format
        
        Args:
            file_path: Source file path
            format_ext: File format extension
            
        Returns:
            Loaded DataFrame
        """
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            if format_ext == 'csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif format_ext == 'json':
                df = pd.read_json(file_path, orient='records')
            elif format_ext == 'excel':
                df = pd.read_excel(file_path, engine='openpyxl')
            elif format_ext == 'parquet':
                try:
                    df = pd.read_parquet(file_path, engine='pyarrow')
                except ImportError:
                    logger.error(f"pyarrow not available, cannot read parquet file {file_path}")
                    return pd.DataFrame()
            else:
                raise ValueError(f"Unsupported format: {format_ext}")
            
            # Normalize datetime columns
            df = self._normalize_datetime_columns(df)
            return df
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return pd.DataFrame()
    
    def insert_dataframe(self, df: pd.DataFrame, schema: str, tb_name: str, 
                        index: List[str], auto_extend_col: bool = False,
                        format_ext: str = 'parquet') -> None:
        """
        Insert/update DataFrame into file storage
        
        Args:
            df: DataFrame to insert
            schema: Schema name (used as directory)
            tb_name: Table name (used as filename base)
            index: List of columns to use as primary key for deduplication
            auto_extend_col: Whether to automatically add new columns (always True for files)
            format_ext: Target file format (default 'parquet'). Must be supported and
                have its dependencies installed. No silent fallback occurs.
        """
        if df is None or df.empty:
            logger.warning(f"Empty DataFrame provided for {schema}.{tb_name}")
            return
        
        # Prepare DataFrame
        df_prepared = self._prepare_dataframe_for_storage(df.copy())
        
        # Handle duplicates based on index columns
        if index and len(index) > 0:
            if df_prepared.duplicated(subset=index).any():
                duplicates = df_prepared[df_prepared.duplicated(subset=index, keep=False)].copy()
                df_prepared = df_prepared.drop_duplicates(subset=index, keep='first')
                logger.warning(f"{schema}.{tb_name} contains duplicates based on index {index}. "
                             f"Duplicates removed: {len(duplicates)} rows")
        
        # Ensure directory exists
        self._ensure_schema_dir(schema, tb_name)
        
        # Determine format to use (explicit parameter overrides autodetect)
        format_to_use = format_ext or self.default_format
        self._validate_format(format_to_use, raise_missing_dependency=True)

        # If a file already exists in a different format than requested, raise to avoid ambiguity
        existing_format = self._detect_existing_format(schema, tb_name)
        if existing_format and existing_format != format_to_use:
            raise ValueError(
                f"Existing table {schema}.{tb_name} stored as {existing_format}; requested format {format_to_use} differs."
            )
        
        file_path = self._get_file_path(schema, tb_name, format_to_use)
        
        # If file exists, merge with existing data based on index
        if file_path.exists() and index:
            existing_df = self._load_dataframe(file_path, format_to_use)
            if not existing_df.empty:
                # Merge dataframes, with new data taking precedence
                # First, remove rows from existing data that match the index of new data
                mask = ~existing_df.set_index(index).index.isin(df_prepared.set_index(index).index)
                existing_df_filtered = existing_df[mask].copy() if mask.any() else pd.DataFrame()
                
                # Combine existing (non-conflicting) and new data
                if not existing_df_filtered.empty:
                    df_prepared = pd.concat([existing_df_filtered, df_prepared], ignore_index=True)
        
        # Save the DataFrame
        self._save_dataframe(df_prepared, file_path, format_to_use)
        logger.info(f"Data inserted into {schema}.{tb_name}: DataFrame{df_prepared.shape}, format: {format_to_use}")
    
    def read_dataframe(self, schema: str, tb_name: str, 
                      filter_column: Optional[List[str]] = None,
                      filter_keyword: Optional[Dict[str, Dict[str, Any]]] = None,
                      ascending: Optional[List[tuple]] = None,
                      limit: Optional[int] = None,
                      format_ext: str = 'parquet') -> pd.DataFrame:
        """
        Read DataFrame from file storage
        
        Args:
            schema: Schema name
            tb_name: Table name
            filter_column: List of columns to select
            filter_keyword: Dictionary of filters (e.g., {'col': {'eq': 'value', 'gt': 10}})
            ascending: List of (column, ascending) tuples for sorting
            limit: Maximum number of rows to return
            format_ext: Expected file format (default 'parquet'). If a different
                on-disk format exists, an error is raised to prevent accidental
                cross-format reads.
            
        Returns:
            DataFrame with requested data
        """
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
        df = self._load_dataframe(file_path, existing_format)
        
        if df.empty:
            return df
        
        # Apply column filtering
        if filter_column:
            available_cols = [col for col in filter_column if col in df.columns]
            if available_cols:
                df = df[available_cols].copy()
            else:
                logger.warning(f"None of the requested columns {filter_column} found in {schema}.{tb_name}")
                return pd.DataFrame()
        
        # Apply filters
        if filter_keyword:
            df = self._apply_filters(df, filter_keyword)
        
        # Apply sorting
        if ascending:
            sort_cols = []
            sort_ascending = []
            for col, asc in ascending:
                if col in df.columns:
                    sort_cols.append(col)
                    sort_ascending.append(asc)
            
            if sort_cols:
                df = df.sort_values(by=sort_cols, ascending=sort_ascending)
        
        # Apply limit
        if limit and limit > 0:
            df = df.head(limit)
        
        return df.reset_index(drop=True)
    
    def _apply_filters(self, df: pd.DataFrame, filter_keyword: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply filters to DataFrame
        
        Args:
            df: Input DataFrame
            filter_keyword: Dictionary of filters
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        for col, filters in filter_keyword.items():
            if col not in filtered_df.columns:
                continue
            
            for op, value in filters.items():
                if op == 'eq':
                    filtered_df = filtered_df[filtered_df[col] == value]
                elif op == 'ne':
                    filtered_df = filtered_df[filtered_df[col] != value]
                elif op == 'gt':
                    filtered_df = filtered_df[filtered_df[col] > value]
                elif op == 'gte':
                    filtered_df = filtered_df[filtered_df[col] >= value]
                elif op == 'lt':
                    filtered_df = filtered_df[filtered_df[col] < value]
                elif op == 'lte':
                    filtered_df = filtered_df[filtered_df[col] <= value]
                elif op == 'in' and isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[col].isin(value)]
                else:
                    logger.warning(f"Unsupported filter operation: {op}")
        
        return filtered_df
    
    def exists(self, schema: str, tb_name: str, filter_dict: Dict[str, Any], format_ext: str = 'parquet') -> bool:
        """
        Check if records matching the filter exist
        
        Args:
            schema: Schema name
            tb_name: Table name
            filter_dict: Dictionary of key-value pairs to filter by
            format_ext: Expected file format (default 'parquet')
            
        Returns:
            True if matching records exist
        """
        if not filter_dict:
            return False
        
        # Convert simple filter dict to filter_keyword format
        filter_keyword = {}
        for key, value in filter_dict.items():
            filter_keyword[key] = {'eq': value}
        
        df = self.read_dataframe(schema, tb_name, filter_keyword=filter_keyword, limit=1, format_ext=format_ext)
        return not df.empty
    
    def delete_records(self, schema: str, tb_name: str, filter_dict: Dict[str, Any], format_ext: str = 'parquet') -> None:
        """
        Delete records matching the filter
        
        Args:
            schema: Schema name
            tb_name: Table name
            filter_dict: Dictionary of key-value pairs to filter by
        """
        requested_format = format_ext or self.default_format
        existing_format = self._detect_existing_format(schema, tb_name)
        if not existing_format:
            logger.warning(f"No file found for {schema}.{tb_name}")
            return
        if existing_format != requested_format:
            raise ValueError(
                f"Existing table {schema}.{tb_name} format {existing_format} does not match requested {requested_format}"
            )

        file_path = self._get_file_path(schema, tb_name, existing_format)
        df = self._load_dataframe(file_path, existing_format)
        
        if df.empty:
            return
        
        # Create filter condition
        mask = pd.Series([True] * len(df))
        for key, value in filter_dict.items():
            if key in df.columns:
                mask = mask & (df[key] == value)
        
        # Keep records that don't match the filter
        df_filtered = df[~mask].copy()
        
        # Save the filtered DataFrame
        self._save_dataframe(df_filtered, file_path, existing_format)
        
        deleted_count = len(df) - len(df_filtered)
        logger.info(f"Deleted {deleted_count} records from {schema}.{tb_name}")
    
    def drop_table(self, schema: str, tb_name: str, format_ext: str = 'parquet') -> None:
        """
        Drop/delete a table (file)
        
        Args:
            schema: Schema name
            tb_name: Table name
        """
        # Only consider the explicitly requested (or default) format to avoid accidental cross-format deletion
        requested_format = format_ext or self.default_format
        existing_format = self._detect_existing_format(schema, tb_name)
        if not existing_format:
            logger.warning(f"No file found to drop for {schema}.{tb_name}")
            return
        if existing_format != requested_format:
            raise ValueError(
                f"Attempted to drop {schema}.{tb_name} as {requested_format} but on-disk format is {existing_format}"
            )
        file_path = self._get_file_path(schema, tb_name, existing_format)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Dropped table {schema}.{tb_name} (format: {existing_format})")
        else:
            logger.warning(f"File path not found for drop: {file_path}")
    
    def get_table_names(self, schema: str) -> List[str]:
        """
        Get list of table names in a schema
        
        Searches both hierarchical (schema/status/country) and traditional (schema) directory structures
        
        Args:
            schema: Schema name
            
        Returns:
            List of table names (reconstructed from hierarchical structure when applicable)
        """
        table_names = set()
        
        # Search hierarchical structure: base_path/schema/status/country/
        schema_path = self.base_path / schema
        if schema_path.exists() and schema_path.is_dir():
            for status_path in schema_path.iterdir():
                if status_path.is_dir() and status_path.name in ['raw', 'processed']:
                    for country_path in status_path.iterdir():
                        if country_path.is_dir() and country_path.name in ['cn', 'us', 'eu']:
                            for file_path in country_path.iterdir():
                                if file_path.is_file():
                                    # Remove file extension to get remaining name
                                    name_parts = file_path.name.rsplit('.', 1)
                                    if len(name_parts) == 2 and name_parts[1] in self.SUPPORTED_FORMATS:
                                        # Reconstruct full table name: status_country_remaining_name
                                        full_name = f"{status_path.name}_{country_path.name}_{name_parts[0]}"
                                        table_names.add(full_name)
        
        # Also search traditional schema structure for backward compatibility
        if schema_path.exists() and schema_path.is_dir():
            for file_path in schema_path.iterdir():
                if file_path.is_file():
                    # Remove file extension to get table name
                    name_parts = file_path.name.rsplit('.', 1)
                    if len(name_parts) == 2 and name_parts[1] in self.SUPPORTED_FORMATS:
                        table_names.add(name_parts[0])
        
        return sorted(list(table_names))
    
    def get_column_names(self, schema: str, tb_name: str, format_ext: str = 'parquet') -> List[str]:
        """
        Get list of column names in a table
        
        Args:
            schema: Schema name
            tb_name: Table name
            
        Returns:
            List of column names
        """
        requested_format = format_ext or self.default_format
        existing_format = self._detect_existing_format(schema, tb_name)
        if not existing_format:
            return []
        if existing_format != requested_format:
            raise ValueError(
                f"Existing table {schema}.{tb_name} format {existing_format} does not match requested {requested_format}"
            )
        file_path = self._get_file_path(schema, tb_name, existing_format)
        df = self._load_dataframe(file_path, existing_format)
        
        return df.columns.tolist()
    
    def insert_json(self, data: Any, schema: str, tb_name: str) -> None:
        """
        Insert/update JSON data into file storage
        
        Args:
            data: JSON-like data to insert (dict, list, or JSON-serializable object)
            schema: Schema name (used as directory/namespace)
            tb_name: Table name (used as filename base)
            
        Note:
            This method performs full-volume writes and does not support 
            incremental updates or deduplication like insert_dataframe.
        """
        if data is None:
            logger.warning(f"None data provided for {schema}.{tb_name}")
            return
        
        # Ensure directory exists
        self._ensure_schema_dir(schema, tb_name)
        
        # Always use JSON format for JSON methods
        format_to_use = 'json'
        self._validate_format(format_to_use, raise_missing_dependency=True)
        
        file_path = self._get_file_path(schema, tb_name, format_to_use)
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON data directly
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON data inserted into {schema}.{tb_name}: {file_path}")
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data to JSON for {schema}.{tb_name}: {e}")
            raise ValueError(f"Data is not JSON-serializable: {e}") from e
        except Exception as e:
            logger.error(f"Failed to write JSON file {file_path}: {e}")
            raise
    
    def read_json(self, schema: str, tb_name: str) -> Any:
        """
        Read JSON data from file storage
        
        Args:
            schema: Schema name
            tb_name: Table name
            
        Returns:
            JSON data (dict, list, or other JSON-serializable object)
            
        Note:
            This method performs full-volume reads and does not support 
            filtering or column selection like read_dataframe.
        """
        # Always use JSON format for JSON methods
        format_to_use = 'json'
        self._validate_format(format_to_use, raise_missing_dependency=True)
        
        file_path = self._get_file_path(schema, tb_name, format_to_use)
        
        if not file_path.exists():
            logger.warning(f"No JSON file found for {schema}.{tb_name}")
            return None
        
        # Load JSON data directly
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"JSON data read from {schema}.{tb_name}: {file_path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in file {file_path}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            raise