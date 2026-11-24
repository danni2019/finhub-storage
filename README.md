# finHub Storage

DataFrame storage extracted from finHub. The package supports local Parquet files, a DuckDB-backed Parquet path, and a PostgreSQL adapter. Configuration is supplied by the caller.

```sh
python -m pip install .
python examples/demo.py
```

The demo uses synthetic prices and a temporary directory. Install `.[duckdb]` or `.[postgres]` for the optional backends. Construct `DuckDBFileStorage` with a storage directory and `PostgreSQL` with a connection mapping. Each backend exposes `insert_dataframe` and `read_dataframe`. Raw SQL helpers take developer-controlled SQL.

The release preserves the 2025 history for the extracted storage files. The packaging commit separates local configuration and licensing from the original application.
