import json
from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

DATA_PATH = Path("data/twcs/twcs.csv")
OUTPUT_PATH = Path("data/apple_conversations.jsonl")


# =========================================================
# STEP 1
# Find all tweet IDs related to AppleSupport
# =========================================================

def collect_relevant_ids():

    print("Step 1: Finding AppleSupport conversations...")

    apple_ids = set()
    relevant_ids = set()

    # Read full dataset in chunks
    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=200_000
    ):

        # AppleSupport tweets
        apple_rows = chunk[
            chunk["author_id"].astype(str) == "AppleSupport"
        ]

        for _, row in apple_rows.iterrows():

            tweet_id = str(row["tweet_id"])

            apple_ids.add(tweet_id)
            relevant_ids.add(tweet_id)

            # Parent tweet
            parent_id = row["in_response_to_tweet_id"]

            if pd.notna(parent_id):
                relevant_ids.add(str(parent_id))

            # Response tweet
            response_id = row["response_tweet_id"]

            if pd.notna(response_id):
                relevant_ids.add(str(response_id))

    print(f"AppleSupport tweets: {len(apple_ids)}")
    print(f"Relevant tweet IDs: {len(relevant_ids)}")

    return apple_ids, relevant_ids


# =========================================================
# STEP 2
# Extract all relevant messages from full dataset
# =========================================================

def extract_relevant_messages(relevant_ids):

    print("\nStep 2: Extracting conversation messages...")

    chunks = []

    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=200_000
    ):

        chunk["tweet_id_str"] = (
            chunk["tweet_id"]
            .astype(str)
        )

        filtered = chunk[
            chunk["tweet_id_str"].isin(relevant_ids)
        ].copy()

        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        print("No relevant messages found.")
        return pd.DataFrame()

    df = pd.concat(
        chunks,
        ignore_index=True
    )

    print(
        f"Relevant messages extracted: {len(df)}"
    )

    return df


# =========================================================
# STEP 3
# Build conversation threads
# =========================================================

def build_conversations(df, apple_ids):

    print("\nStep 3: Building conversation threads...")

    if df.empty:
        return []

    # -----------------------------------------------------
    # Create tweet lookup
    # -----------------------------------------------------

    tweet_map = {}

    for _, row in df.iterrows():

        tweet_id = str(row["tweet_id"])

        tweet_map[tweet_id] = {
            "tweet_id": tweet_id,
            "author_id": str(row["author_id"]),
            "inbound": row["inbound"],
            "created_at": str(row["created_at"]),
            "text": str(row["text"]),
            "response_tweet_id": (
                ""
                if pd.isna(row["response_tweet_id"])
                else str(row["response_tweet_id"])
            ),
            "in_response_to_tweet_id": (
                ""
                if pd.isna(row["in_response_to_tweet_id"])
                else str(row["in_response_to_tweet_id"])
            )
        }

    print(f"Indexed tweets: {len(tweet_map)}")

    # -----------------------------------------------------
    # Build graph
    # -----------------------------------------------------

    graph = {
        tweet_id: set()
        for tweet_id in tweet_map
    }

    for tweet_id, tweet in tweet_map.items():

        parent_id = tweet["in_response_to_tweet_id"]
        response_id = tweet["response_tweet_id"]

        if parent_id in tweet_map:

            graph[tweet_id].add(parent_id)
            graph[parent_id].add(tweet_id)

        if response_id in tweet_map:

            graph[tweet_id].add(response_id)
            graph[response_id].add(tweet_id)

    # -----------------------------------------------------
    # Connected components
    # -----------------------------------------------------

    visited = set()
    conversations = []

    for tweet_id in tweet_map:

        if tweet_id in visited:
            continue

        stack = [tweet_id]
        component = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            for neighbour in graph[current]:

                if neighbour not in visited:
                    stack.append(neighbour)

        messages = [
            tweet_map[x]
            for x in component
        ]

        # -------------------------------------------------
        # Must contain AppleSupport
        # -------------------------------------------------

        has_apple = any(
            message["tweet_id"] in apple_ids
            for message in messages
        )

        if not has_apple:
            continue

        # -------------------------------------------------
        # Sort chronologically
        # -------------------------------------------------

        messages.sort(
            key=lambda x: x["created_at"]
        )

        # -------------------------------------------------
        # Remove empty text
        # -------------------------------------------------

        messages = [
            message
            for message in messages
            if message["text"].strip()
        ]

        if len(messages) < 2:
            continue

        # -------------------------------------------------
        # Convert to customer / agent
        #
        # inbound=True  -> customer
        # inbound=False -> AppleSupport
        # -------------------------------------------------

        formatted_messages = []

        for message in messages:

            inbound = message["inbound"]

            if str(inbound).lower() == "true":
                role = "customer"
            else:
                role = "agent"

            formatted_messages.append({
                "tweet_id": message["tweet_id"],
                "role": role,
                "author_id": message["author_id"],
                "created_at": message["created_at"],
                "text": message["text"]
            })

        # -------------------------------------------------
        # Create conversation
        # -------------------------------------------------

        conversation = {
            "conversation_id": (
                f"apple_conv_{len(conversations) + 1:06d}"
            ),
            "message_count": len(formatted_messages),
            "messages": formatted_messages
        }

        conversations.append(conversation)

    print(
        f"Total conversations: {len(conversations)}"
    )

    return conversations


# =========================================================
# STEP 4
# Save JSONL
# =========================================================

def save_conversations(conversations):

    print("\nStep 4: Saving conversations...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for conversation in conversations:

            f.write(
                json.dumps(
                    conversation,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


# =========================================================
# STEP 5
# Show sample
# =========================================================

def show_sample(conversations):

    print("\n" + "=" * 70)
    print("SAMPLE CONVERSATION")
    print("=" * 70)

    if not conversations:

        print("WARNING: No conversations found.")
        return

    conversation = conversations[0]

    print(
        f"Conversation ID: "
        f"{conversation['conversation_id']}"
    )

    print(
        f"Messages: "
        f"{conversation['message_count']}"
    )

    print()

    for message in conversation["messages"]:

        print(
            f"[{message['role'].upper()}] "
            f"{message['text']}"
        )

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("AppleSupport Conversation Builder")
    print("=" * 70)

    # 1. Find AppleSupport + related tweet IDs
    apple_ids, relevant_ids = collect_relevant_ids()

    # 2. Get customer + agent messages from FULL dataset
    df = extract_relevant_messages(
        relevant_ids
    )

    # 3. Build threads
    conversations = build_conversations(
        df,
        apple_ids
    )

    # 4. Save
    save_conversations(
        conversations
    )

    # 5. Show example
    show_sample(
        conversations
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()