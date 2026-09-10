import json
import random
import csv
from pathlib import Path

DATA_PATH = Path("data/apple_conversations.jsonl")
OUTPUT_PATH = Path("data/golden_sample.csv")

SAMPLE_SIZE = 200
RANDOM_SEED = 42
CONTEXT_MESSAGES = 4


def load_conversations():
    conversations = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            conversations.append(json.loads(line))

    return conversations


def build_candidates(conversations):
    candidates = []

    for conversation in conversations:

        messages = conversation.get("messages", [])

        for index, message in enumerate(messages):

            # We need customer messages
            if message.get("role") != "customer":
                continue

            customer_text = message.get("text", "").strip()

            if not customer_text:
                continue

            # Find the next agent response
            next_agent_reply = None

            for next_index in range(index + 1, len(messages)):

                if messages[next_index].get("role") == "agent":

                    reply = messages[next_index].get("text", "").strip()

                    if reply:
                        next_agent_reply = reply

                    break

            # For reply-generation evaluation,
            # keep examples that have a historical agent response.
            if not next_agent_reply:
                continue

            # Previous conversation context
            start_index = max(0, index - CONTEXT_MESSAGES)

            previous_messages = messages[start_index:index]

            context_parts = []

            for previous in previous_messages:

                role = previous.get("role", "").upper()
                text = previous.get("text", "").strip()

                if text:
                    context_parts.append(f"{role}: {text}")

            context = "\n".join(context_parts)

            candidates.append({
                "conversation_id": conversation["conversation_id"],
                "customer_message": customer_text,
                "context": context,
                "historical_agent_reply": next_agent_reply
            })

    return candidates


def sample_examples(candidates):

    random.seed(RANDOM_SEED)

    # Avoid selecting multiple messages from the same conversation.
    by_conversation = {}

    for candidate in candidates:

        conversation_id = candidate["conversation_id"]

        if conversation_id not in by_conversation:
            by_conversation[conversation_id] = []

        by_conversation[conversation_id].append(candidate)

    conversation_ids = list(by_conversation.keys())

    random.shuffle(conversation_ids)

    selected = []

    for conversation_id in conversation_ids:

        if len(selected) >= SAMPLE_SIZE:
            break

        examples = by_conversation[conversation_id]

        selected.append(random.choice(examples))

    return selected


def save_csv(samples):

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "example_id",
        "conversation_id",
        "customer_message",
        "context",
        "historical_agent_reply",
        "intent",
        "escalation",
        "notes"
    ]

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index, sample in enumerate(samples, start=1):

            writer.writerow({
                "example_id": f"gold_{index:03d}",
                "conversation_id": sample["conversation_id"],
                "customer_message": sample["customer_message"],
                "context": sample["context"],
                "historical_agent_reply": sample["historical_agent_reply"],

                # These are intentionally blank.
                # We will manually label them.
                "intent": "",
                "escalation": "",
                "notes": ""
            })


def print_preview(samples):

    print("\n" + "=" * 70)
    print("GOLDEN SAMPLE CREATED")
    print("=" * 70)

    print(f"\nEligible customer turns: {len(samples)}")
    print(f"Selected examples: {len(samples)}")

    print("\nFirst 5 examples:\n")

    for sample in samples[:5]:

        print("-" * 70)

        print("Conversation:", sample["conversation_id"])

        print("Customer:")
        print(sample["customer_message"])

        if sample["context"]:
            print("\nPrevious context:")
            print(sample["context"])

        print("\nHistorical AppleSupport reply:")
        print(sample["historical_agent_reply"])


def main():

    print("=" * 70)
    print("APPLE SUPPORT — GOLDEN SAMPLE CREATION")
    print("=" * 70)

    print("\nLoading conversations...")

    conversations = load_conversations()

    print(f"Conversations loaded: {len(conversations)}")

    print("\nFinding suitable customer turns...")

    candidates = build_candidates(conversations)

    print(f"Eligible customer turns: {len(candidates)}")

    print("\nSampling examples...")

    samples = sample_examples(candidates)

    print(f"Selected examples: {len(samples)}")

    print("\nSaving CSV...")

    save_csv(samples)

    print_preview(samples)

    print("\n" + "=" * 70)
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 70)

    print("\nDONE.")


if __name__ == "__main__":
    main()