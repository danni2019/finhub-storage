# finHub Storage

Store and retrieve pandas DataFrames using local Parquet files, DuckDB over Parquet, or PostgreSQL. Supply the storage path or connection mapping from your application.

## Try a Parquet round trip

From a checkout with Python 3.11 or newer:

```sh
python -m pip install .
python examples/demo.py
```

The [demo](examples/demo.py) writes three synthetic prices to a temporary directory and reads rows where `id >= 2`. It prints:

```text
 id  price
  2  102.5
  3  103.0
```

| Backend | Use it for | Installation |
| --- | --- | --- |
| `LocalFileStorage` | Local Parquet reads and writes | `python -m pip install .` |
| `DuckDBFileStorage` | DuckDB-backed filtering and upserts on local files | `python -m pip install '.[duckdb]'` |
| `PostgreSQL` | DataFrame reads and writes against a PostgreSQL server you configure | `python -m pip install '.[postgres]'` |

Each backend exposes `insert_dataframe` and `read_dataframe`. Raw SQL helpers take developer-controlled SQL.

This [MIT-licensed extraction](LICENSE) preserves the related 2025 storage history.
