import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs/twcs.csv")
OUTPUT_PATH = Path("data/apple_support.csv")

BRAND = "AppleSupport"


def extract_brand_data():
    apple_chunks = []

    print(f"Extracting {BRAND} conversations...")

    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=200_000
    ):
        apple_rows = chunk[
            chunk["author_id"].astype(str).eq(BRAND)
        ]

        if not apple_rows.empty:
            apple_chunks.append(apple_rows)

    apple_df = pd.concat(apple_chunks, ignore_index=True)

    apple_df.to_csv(OUTPUT_PATH, index=False)

    print("\nExtraction complete!")
    print("AppleSupport rows:", len(apple_df))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    extract_brand_data()