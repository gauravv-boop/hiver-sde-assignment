import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs/twcs.csv")


def inspect_dataset():
    print("Loading first 10,000 rows...")

    df = pd.read_csv(DATA_PATH, nrows=10_000)

    print("\nRows:", len(df))
    print("\nColumns:")
    for column in df.columns:
        print("-", column)

    print("\nSample:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    inspect_dataset()