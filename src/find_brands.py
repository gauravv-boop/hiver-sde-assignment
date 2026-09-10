import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs/twcs.csv")


def find_brand_accounts():
    brand_counts = {}

    print("Scanning dataset for brand accounts...")

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=["author_id", "inbound"],
        chunksize=200_000
    ):
        # Brand responses have inbound=False
        brand_messages = chunk[chunk["inbound"] == False]

        counts = brand_messages["author_id"].value_counts()

        for author_id, count in counts.items():
            brand_counts[author_id] = (
                brand_counts.get(author_id, 0) + count
            )

    results = sorted(
        brand_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop brand accounts:\n")

    for author_id, count in results[:30]:
        print(f"{author_id}: {count}")


if __name__ == "__main__":
    find_brand_accounts()