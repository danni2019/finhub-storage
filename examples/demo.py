from tempfile import TemporaryDirectory

import pandas as pd

from finhub_storage.local_file_storage import LocalFileStorage


def main():
    prices = pd.DataFrame({"id": [1, 2, 3], "price": [101.0, 102.5, 103.0]})
    with TemporaryDirectory() as directory:
        storage = LocalFileStorage({"base_path": directory})
        storage.insert_dataframe(prices, "market", "raw_demo_prices", index=["id"])
        found = storage.read_dataframe(
            "market", "raw_demo_prices", filter_keyword={"id": {"gte": 2}}
        )
        print(found.to_string(index=False))


if __name__ == "__main__":
    main()
