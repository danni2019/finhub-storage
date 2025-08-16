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
    DEFAULT_FORMAT = 'csv'  # CSV is most compatible
    
    def __init_class_formats__(self):
        """Initialize supported formats based on available libraries"""
        try:
            import pyarrow
            if 'parquet' not in self.SUPPORTED_FORMATS:
                self.SUPPORTED_FORMATS.append('parquet')
        except ImportError:
            logger.info("pyarrow not available, parquet format disabled")
            if self.default_format == 'parquet':
                self.default_format = 'csv'
    
    def __init__(self, storage_config: dict = None):
        """
        Initialize local file storage
        
        Args:
            storage_config: Configuration dictionary containing:
                - base_path: Base directory for storage (default: ./doc/fs)
                - default_format: Default file format (default: csv)
                - create_dirs: Whether to create directories automatically (default: True)
        """
        self.config = storage_config or {}
        self.base_path = Path(self.config.get('base_path', './doc/fs'))
        self.default_format = self.config.get('default_format', self.DEFAULT_FORMAT)
        self.create_dirs = self.config.get('create_dirs', True)
        
        # Initialize supported formats based on available libraries
        self.__init_class_formats__()
        
        # Ensure base path exists
        if self.create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"LocalFileStorage initialized with base_path: {self.base_path}")
    
    def _get_file_path(self, schema: str, tb_name: str, format_ext: str = None) -> Path:
        """
        Get the full file path for a table
        
        Args:
            schema: Schema name (directory)
            tb_name: Table name (filename base)
            format_ext: File format extension (default: self.default_format)
            
        Returns:
            Path object for the file
        """
        if format_ext is None:
            format_ext = self.default_format
        
        schema_path = self.base_path / schema
        file_path = schema_path / f"{tb_name}.{format_ext}"
        
        return file_path
    
    def _ensure_schema_dir(self, schema: str) -> None:
        """
        Ensure schema directory exists
        
        Args:
            schema: Schema name
        """
        schema_path = self.base_path / schema
        if self.create_dirs:
            schema_path.mkdir(parents=True, exist_ok=True)
    
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
            try:
                df.to_parquet(file_path, index=False, engine='pyarrow')
            except ImportError:
                logger.warning(f"pyarrow not available, falling back to CSV for {file_path}")
                # Change extension to csv and save as CSV
                csv_path = file_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8')
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
                        index: List[str], auto_extend_col: bool = False) -> None:
        """
        Insert/update DataFrame into file storage
        
        Args:
            df: DataFrame to insert
            schema: Schema name (used as directory)
            tb_name: Table name (used as filename base)
            index: List of columns to use as primary key for deduplication
            auto_extend_col: Whether to automatically add new columns (always True for files)
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
        
        # Ensure schema directory exists
        self._ensure_schema_dir(schema)
        
        # Detect existing file format or use default
        existing_format = self._detect_existing_format(schema, tb_name)
        format_to_use = existing_format or self.default_format
        
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
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Read DataFrame from file storage
        
        Args:
            schema: Schema name
            tb_name: Table name
            filter_column: List of columns to select
            filter_keyword: Dictionary of filters (e.g., {'col': {'eq': 'value', 'gt': 10}})
            ascending: List of (column, ascending) tuples for sorting
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with requested data
        """
        # Detect existing file format
        existing_format = self._detect_existing_format(schema, tb_name)
        if not existing_format:
            logger.warning(f"No file found for {schema}.{tb_name}")
            return pd.DataFrame()
        
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
    
    def exists(self, schema: str, tb_name: str, filter_dict: Dict[str, Any]) -> bool:
        """
        Check if records matching the filter exist
        
        Args:
            schema: Schema name
            tb_name: Table name
            filter_dict: Dictionary of key-value pairs to filter by
            
        Returns:
            True if matching records exist
        """
        if not filter_dict:
            return False
        
        # Convert simple filter dict to filter_keyword format
        filter_keyword = {}
        for key, value in filter_dict.items():
            filter_keyword[key] = {'eq': value}
        
        df = self.read_dataframe(schema, tb_name, filter_keyword=filter_keyword, limit=1)
        return not df.empty
    
    def delete_records(self, schema: str, tb_name: str, filter_dict: Dict[str, Any]) -> None:
        """
        Delete records matching the filter
        
        Args:
            schema: Schema name
            tb_name: Table name
            filter_dict: Dictionary of key-value pairs to filter by
        """
        existing_format = self._detect_existing_format(schema, tb_name)
        if not existing_format:
            logger.warning(f"No file found for {schema}.{tb_name}")
            return
        
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
    
    def drop_table(self, schema: str, tb_name: str) -> None:
        """
        Drop/delete a table (file)
        
        Args:
            schema: Schema name
            tb_name: Table name
        """
        # Try to find and delete file in any supported format
        deleted = False
        for fmt in self.SUPPORTED_FORMATS:
            file_path = self._get_file_path(schema, tb_name, fmt)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Dropped table {schema}.{tb_name} (format: {fmt})")
                deleted = True
        
        if not deleted:
            logger.warning(f"No file found to drop for {schema}.{tb_name}")
    
    def get_table_names(self, schema: str) -> List[str]:
        """
        Get list of table names in a schema
        
        Args:
            schema: Schema name
            
        Returns:
            List of table names (without file extensions)
        """
        schema_path = self.base_path / schema
        if not schema_path.exists() or not schema_path.is_dir():
            return []
        
        table_names = set()
        for file_path in schema_path.iterdir():
            if file_path.is_file():
                # Remove file extension to get table name
                name_parts = file_path.name.rsplit('.', 1)
                if len(name_parts) == 2 and name_parts[1] in self.SUPPORTED_FORMATS:
                    table_names.add(name_parts[0])
        
        return sorted(list(table_names))
    
    def get_column_names(self, schema: str, tb_name: str) -> List[str]:
        """
        Get list of column names in a table
        
        Args:
            schema: Schema name
            tb_name: Table name
            
        Returns:
            List of column names
        """
        existing_format = self._detect_existing_format(schema, tb_name)
        if not existing_format:
            return []
        
        file_path = self._get_file_path(schema, tb_name, existing_format)
        df = self._load_dataframe(file_path, existing_format)
        
        return df.columns.tolist()