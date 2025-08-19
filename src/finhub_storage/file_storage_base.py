import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Dict, Any
from datetime import datetime


class FileStorageBase(ABC):
    """
    Abstract base class for file storage systems
    
    This class provides a common interface for different file storage implementations
    including local file storage and future online storage systems like ArcticDB.
    
    The interface is designed to be compatible with the existing PostgreSQL interface,
    maintaining the same method signatures for insert_dataframe and read_dataframe.
    """
    
    @abstractmethod
    def __init__(self, storage_config: dict = None):
        """
        Initialize the file storage system
        
        Args:
            storage_config: Configuration dictionary for the storage system
        """
        pass
    
    @abstractmethod
    def insert_dataframe(self, df: pd.DataFrame, schema: str, tb_name: str, 
                        index: List[str], auto_extend_col: bool = False) -> None:
        """
        Insert/update DataFrame into file storage
        
        Args:
            df: DataFrame to insert
            schema: Schema name (used as directory/namespace)
            tb_name: Table name (used as filename base)
            index: List of columns to use as primary key for deduplication
            auto_extend_col: Whether to automatically add new columns
        """
        pass
    
    @abstractmethod
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
            filter_keyword: Dictionary of filters
            ascending: List of (column, ascending) tuples for sorting
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with requested data
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def delete_records(self, schema: str, tb_name: str, 
                      filter_dict: Dict[str, Any]) -> None:
        """
        Delete records matching the filter
        
        Args:
            schema: Schema name
            tb_name: Table name
            filter_dict: Dictionary of key-value pairs to filter by
        """
        pass
    
    @abstractmethod
    def drop_table(self, schema: str, tb_name: str) -> None:
        """
        Drop/delete a table (file)
        
        Args:
            schema: Schema name
            tb_name: Table name
        """
        pass
    
    @abstractmethod
    def get_table_names(self, schema: str) -> List[str]:
        """
        Get list of table names in a schema
        
        Args:
            schema: Schema name
            
        Returns:
            List of table names
        """
        pass
    
    @abstractmethod
    def get_column_names(self, schema: str, tb_name: str) -> List[str]:
        """
        Get list of column names in a table
        
        Args:
            schema: Schema name
            tb_name: Table name
            
        Returns:
            List of column names
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @staticmethod
    def _normalize_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize datetime columns following the same rules as PostgreSQL implementation
        
        Columns containing 'date' or 'time' in name are converted to datetime64[ns]
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized datetime columns
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        for col in df_copy.columns:
            col_lower = str(col).lower()
            if ('date' in col_lower) or ('time' in col_lower):
                try:
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                except Exception:
                    # If conversion fails, leave as is
                    pass
        
        return df_copy
    
    @staticmethod
    def _prepare_dataframe_for_storage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for storage by handling data types and NaN values
        
        Similar to PostgreSQL's __alter_df_type_for_sql__ method
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame ready for storage
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        for col in df_copy.columns:
            col_lower = str(col).lower()
            dtype_str = str(df_copy[col].dtype).lower()
            
            # Handle datetime columns
            if ('date' in col_lower) or ('time' in col_lower):
                series = pd.to_datetime(df_copy[col], errors='coerce')
                # Remove timezone info if present (similar to PostgreSQL implementation)
                try:
                    if isinstance(series.dtype, pd.DatetimeTZDtype):
                        series = series.dt.tz_localize(None)
                except Exception:
                    series = series.apply(
                        lambda x: x.replace(tzinfo=None) if (pd.notna(x) and getattr(x, 'tzinfo', None)) else x
                    )
                df_copy[col] = series
            
            # Handle NaN values in numeric columns
            elif 'int' in dtype_str:
                # For integer columns with NaN, convert to nullable integer
                if df_copy[col].isna().any():
                    df_copy[col] = df_copy[col].astype('Int64')
            elif 'float' in dtype_str:
                # Float columns can naturally handle NaN
                df_copy[col] = df_copy[col].astype('float64')
        
        return df_copy