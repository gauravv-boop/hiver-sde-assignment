import json
import re
from pathlib import Path
from collections import Counter

DATA_PATH = Path("data/apple_conversations.jsonl")
OUTPUT_PATH = Path("data/intent_distribution.txt")


# --------------------------------------------------
# INTENT DEFINITIONS
# --------------------------------------------------

INTENT_KEYWORDS = {
    "account_login_verification": [
        "password",
        "login",
        "account",
        "verification",
        "verify",
        "2fa",
        "apple id",
        "recovery",
    ],

    "app_store_app_issue": [
        "app store",
        "app",
        "apps",
        "download",
        "install",
        "update apps",
    ],

    "ios_software_issue": [
        "ios",
        "software",
        "update",
        "updated",
        "upgrade",
        "version",
    ],

    "battery_charging_issue": [
        "battery",
        "charge",
        "charging",
        "charger",
    ],

    "wifi_network_issue": [
        "wifi",
        "wi-fi",
        "network",
        "internet",
        "connection",
        "4g",
    ],

    "icloud_backup_restore": [
        "icloud",
        "backup",
        "restore",
        "restored",
        "sync",
        "data",
    ],

    "billing_payment_purchase": [
        "payment",
        "billing",
        "charged",
        "charge",
        "refund",
        "purchase",
        "credit card",
        "card",
        "paypal",
    ],

    "subscription_media_issue": [
        "subscription",
        "apple music",
        "music",
        "itunes",
    ],

    "device_hardware_issue": [
        "screen",
        "camera",
        "broken",
        "touch",
        "hardware",
    ],

    "messaging_calling_issue": [
        "imessage",
        "message",
        "messages",
        "facetime",
        "call",
        "calling",
        "notification",
    ],
}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_customer_messages():
    messages = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            conversation = json.loads(line)

            for message in conversation.get("messages", []):
                if message.get("role") != "customer":
                    continue

                text = message.get("text", "").strip()

                if text:
                    messages.append(text)

    return messages


# --------------------------------------------------
# KEYWORD MATCHING
# --------------------------------------------------

def contains_keyword(text, keyword):
    text = text.lower()

    keyword = keyword.lower()

    if " " in keyword:
        return keyword in text

    pattern = rf"\b{re.escape(keyword)}\b"

    return re.search(pattern, text) is not None


def classify_message(text):
    matched_intents = []

    for intent, keywords in INTENT_KEYWORDS.items():

        for keyword in keywords:

            if contains_keyword(text, keyword):
                matched_intents.append(intent)
                break

    # No matching intent
    if not matched_intents:
        return "other_unclear"

    # Multiple intents
    # For now, keep the first one.
    # We will inspect multi-intent messages later.
    return matched_intents[0]


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

def analyze(messages):

    intent_counts = Counter()

    examples = {}

    multi_intent_count = 0

    for text in messages:

        matched = []

        for intent, keywords in INTENT_KEYWORDS.items():

            if any(contains_keyword(text, keyword)
                   for keyword in keywords):

                matched.append(intent)

        if len(matched) > 1:
            multi_intent_count += 1

        intent = matched[0] if matched else "other_unclear"

        intent_counts[intent] += 1

        if intent not in examples:
            examples[intent] = []

        if len(examples[intent]) < 5:
            examples[intent].append(text)

    return intent_counts, examples, multi_intent_count


# --------------------------------------------------
# REPORT
# --------------------------------------------------

def write_report(messages, intent_counts, examples, multi_intent_count):

    total = len(messages)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        file.write("APPLE SUPPORT — INTENT DISTRIBUTION\n")
        file.write("=" * 70 + "\n\n")

        file.write(f"Customer messages: {total}\n")
        file.write(f"Messages matching multiple intents: {multi_intent_count}\n\n")

        file.write("INTENT COUNTS\n")
        file.write("-" * 70 + "\n")

        for intent, count in intent_counts.most_common():

            percentage = (count / total) * 100

            file.write(
                f"{intent:35} {count:6} ({percentage:5.2f}%)\n"
            )

        file.write("\n\nEXAMPLES\n")
        file.write("=" * 70 + "\n")

        for intent in intent_counts:

            file.write(f"\n### {intent}\n")

            for index, example in enumerate(
                examples[intent],
                start=1
            ):

                file.write(f"{index}. {example}\n")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 70)
    print("APPLE SUPPORT — INTENT DISTRIBUTION")
    print("=" * 70)

    print("\nLoading customer messages...")

    messages = load_customer_messages()

    print(f"Customer messages: {len(messages)}")

    print("\nClassifying messages using preliminary keyword rules...")

    intent_counts, examples, multi_intent_count = analyze(messages)

    print("\nIntent distribution:\n")

    for intent, count in intent_counts.most_common():

        percentage = (count / len(messages)) * 100

        print(
            f"{intent:35} "
            f"{count:6} "
            f"({percentage:5.2f}%)"
        )

    print(
        f"\nMessages matching multiple intents: "
        f"{multi_intent_count}"
    )

    print("\nWriting report...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_report(
        messages,
        intent_counts,
        examples,
        multi_intent_count
    )

    print(f"\nReport saved to: {OUTPUT_PATH}")

    print("\nDONE.")


if __name__ == "__main__":
    main()