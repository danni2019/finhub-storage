import os

import pandas as pd
import pytest

from finhub_storage.local_file_storage import LocalFileStorage


def exercise_storage(storage):
    source = pd.DataFrame({"id": [1, 2, 3], "price": [100.0, 101.0, 102.0]})
    storage.insert_dataframe(source, "market", "raw_demo_prices", index=["id"])
    new_row = pd.DataFrame({"id": [2], "price": [105.0]})
    storage.insert_dataframe(new_row, "market", "raw_demo_prices", index=["id"])
    result = storage.read_dataframe(
        "market", "raw_demo_prices", filter_keyword={"id": {"gte": 2}}
    )
    assert result.sort_values("id")["price"].tolist() == [105.0, 102.0]


def test_parquet_roundtrip(tmp_path):
    storage = LocalFileStorage({"base_path": tmp_path})
    exercise_storage(storage)
    with pytest.raises(ValueError):
        storage.read_dataframe("..", "raw_demo_prices")


def test_duckdb_roundtrip(tmp_path):
    pytest.importorskip("duckdb")
    from finhub_storage.duckdb_file_storage import DuckDBFileStorage

    exercise_storage(DuckDBFileStorage({"base_path": tmp_path}))


def test_postgres_roundtrip():
    port = os.environ.get("FINHUB_TEST_PG_PORT")
    if not port:
        pytest.skip("isolated integration database was not configured")
    from finhub_storage.pgsql import PostgreSQL

    storage = PostgreSQL({
        "host": "127.0.0.1",
        "port": int(port),
        "dbname": "finhub_test",
        "user": "finhub_test",
        "password": os.environ["FINHUB_TEST_PG_PASSWORD"],
    })
    try:
        storage.drop_table("market", "raw_demo_prices")
        exercise_storage(storage)
    finally:
        storage.drop_table("market", "raw_demo_prices")
        storage.close()
