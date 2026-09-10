import json
import re
from pathlib import Path
from collections import Counter


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = Path("data/apple_conversations.jsonl")
OUTPUT_PATH = Path("data/intent_discovery.txt")

MAX_EXAMPLES_PER_KEYWORD = 5
TOP_KEYWORDS = 100


# =========================================================
# LOAD CONVERSATIONS
# =========================================================

def load_conversations():
    conversations = []

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            conversations.append(
                json.loads(line)
            )

    return conversations


# =========================================================
# EXTRACT CUSTOMER MESSAGES
# =========================================================

def get_customer_messages(conversations):

    messages = []

    for conversation in conversations:

        for message in conversation.get(
            "messages",
            []
        ):

            if message.get("role") != "customer":
                continue

            text = message.get(
                "text",
                ""
            ).strip()

            if text:
                messages.append(text)

    return messages


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Convert HTML entities
    text = text.replace(
        "&amp;",
        "and"
    )

    # Keep alphabetic words
    words = re.findall(
        r"[a-z]{3,}",
        text
    )

    return words


# =========================================================
# STOP WORDS
# =========================================================

STOP_WORDS = {
    "the",
    "and",
    "for",
    "you",
    "this",
    "that",
    "with",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "but",
    "not",
    "can",
    "could",
    "would",
    "should",
    "your",
    "from",
    "they",
    "them",
    "their",
    "just",
    "what",
    "when",
    "where",
    "why",
    "how",
    "please",
    "help",
    "apple",
    "support",
    "need",
    "needs",
    "want",
    "wanted",
    "about",
    "there",
    "been",
    "being",
    "its",
    "it's",
    "im",
    "i'm",
    "into",
    "will",
    "my",
    "me",
    "we",
    "our",
    "get",
    "got",
    "getting",
    "thing",
    "things",
    "still",
    "also",
    "very",
    "much",
    "really",
    "only",
    "now",
    "then",
    "than",
    "some",
    "something",
    "someone",
    "already",
    "because",
    "after",
    "before",
    "again",
    "even",
    "one",
    "two",
    "all",
    "any",
    "more",
    "too",
    "does",
    "doesnt",
    "did",
    "didnt",
    "dont",
    "isnt",
    "cant",
    "wont"
}


# =========================================================
# WORD FREQUENCY
# =========================================================

def get_word_frequency(messages):

    counter = Counter()

    for message in messages:

        words = normalize_text(message)

        for word in words:

            if word not in STOP_WORDS:
                counter[word] += 1

    return counter


# =========================================================
# EXACT KEYWORD MATCHING
# =========================================================

def contains_keyword(text, keyword):

    pattern = rf"\b{re.escape(keyword.lower())}\b"

    return re.search(
        pattern,
        text.lower()
    ) is not None


# =========================================================
# FIND REPRESENTATIVE EXAMPLES
# =========================================================

def find_examples(
    messages,
    keyword,
    limit=MAX_EXAMPLES_PER_KEYWORD
):

    examples = []

    seen = set()

    for message in messages:

        if not contains_keyword(
            message,
            keyword
        ):
            continue

        # Avoid duplicate examples
        normalized = message.lower().strip()

        if normalized in seen:
            continue

        seen.add(normalized)

        examples.append(message)

        if len(examples) >= limit:
            break

    return examples


# =========================================================
# SUPPORT KEYWORDS
# =========================================================

SUPPORT_KEYWORDS = [
    "password",
    "login",
    "account",
    "icloud",
    "iphone",
    "ipad",
    "mac",
    "itunes",
    "app",
    "apps",
    "update",
    "updated",
    "ios",
    "payment",
    "charge",
    "charged",
    "refund",
    "subscription",
    "billing",
    "purchase",
    "order",
    "credit",
    "card",
    "music",
    "download",
    "activation",
    "locked",
    "verification",
    "email",
    "notification",
    "wifi",
    "battery",
    "restore",
    "backup",
    "data",
    "store",
    "screen",
    "keyboard",
    "network",
    "connection",
    "sync",
    "storage",
    "camera",
    "message",
    "messages",
    "imessage",
    "facetime",
    "icloud",
    "developer"
]


# =========================================================
# WRITE REPORT
# =========================================================

def write_report(
    conversations,
    messages,
    frequencies
):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    top_words = frequencies.most_common(
        TOP_KEYWORDS
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as output:

        # -------------------------------------------------
        # Header
        # -------------------------------------------------

        output.write(
            "APPLE SUPPORT — INTENT DISCOVERY\n"
        )

        output.write(
            "=" * 70 + "\n\n"
        )

        output.write(
            f"Conversations: "
            f"{len(conversations)}\n"
        )

        output.write(
            f"Customer messages: "
            f"{len(messages)}\n\n"
        )

        # -------------------------------------------------
        # Top keywords
        # -------------------------------------------------

        output.write(
            "TOP KEYWORDS\n"
        )

        output.write(
            "-" * 70 + "\n"
        )

        for index, (word, count) in enumerate(
            top_words,
            start=1
        ):

            output.write(
                f"{index:3}. "
                f"{word:25} "
                f"{count}\n"
            )

        # -------------------------------------------------
        # Keyword examples
        # -------------------------------------------------

        output.write(
            "\n\n"
        )

        output.write(
            "SUPPORT KEYWORD EXAMPLES\n"
        )

        output.write(
            "=" * 70 + "\n"
        )

        for keyword in SUPPORT_KEYWORDS:

            examples = find_examples(
                messages,
                keyword
            )

            if not examples:
                continue

            output.write(
                f"\n### {keyword.upper()}\n"
            )

            for index, example in enumerate(
                examples,
                start=1
            ):

                output.write(
                    f"{index}. {example}\n"
                )

    return top_words


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    conversations,
    messages,
    top_words
):

    print("\n" + "=" * 70)
    print("INTENT DISCOVERY COMPLETE")
    print("=" * 70)

    print(
        f"Conversations: "
        f"{len(conversations)}"
    )

    print(
        f"Customer messages: "
        f"{len(messages)}"
    )

    print("\nTop keywords:\n")

    for index, (word, count) in enumerate(
        top_words[:30],
        start=1
    ):

        print(
            f"{index:2}. "
            f"{word:20} "
            f"{count}"
        )

    print(
        f"\nDetailed report saved to:"
    )

    print(
        OUTPUT_PATH
    )


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze():

    print("=" * 70)
    print("APPLE SUPPORT — INTENT DISCOVERY")
    print("=" * 70)

    print("\nLoading conversations...")

    conversations = load_conversations()

    print(
        f"Loaded {len(conversations)} conversations."
    )

    print(
        "\nExtracting customer messages..."
    )

    messages = get_customer_messages(
        conversations
    )

    print(
        f"Extracted {len(messages)} customer messages."
    )

    print(
        "\nCalculating keyword frequencies..."
    )

    frequencies = get_word_frequency(
        messages
    )

    print(
        "\nWriting analysis report..."
    )

    top_words = write_report(
        conversations,
        messages,
        frequencies
    )

    print_summary(
        conversations,
        messages,
        top_words
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    analyze()