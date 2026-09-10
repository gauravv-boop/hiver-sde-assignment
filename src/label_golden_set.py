import csv
from pathlib import Path

INPUT_PATH = Path("data/golden_sample.csv")
OUTPUT_PATH = Path("data/golden_sample.csv")


INTENTS = [
    (
        "account_login_verification",
        "Apple ID / account / password / login / 2FA / verification / recovery"
    ),
    (
        "app_store_app_issue",
        "App Store / app download / install / app update / app not working"
    ),
    (
        "ios_software_issue",
        "iOS / software update / system bugs / freezes / update-related problems"
    ),
    (
        "battery_charging_issue",
        "battery drain / charging / charger / phone not charging"
    ),
    (
        "wifi_network_issue",
        "Wi-Fi / internet / network / connectivity"
    ),
    (
        "icloud_backup_restore",
        "iCloud / backup / restore / data recovery / sync"
    ),
    (
        "billing_payment_purchase",
        "billing / payment / card / charge / refund / purchase"
    ),
    (
        "subscription_media_issue",
        "Apple Music / subscription / iTunes / media"
    ),
    (
        "device_hardware_issue",
        "screen / camera / touch / physical device or hardware problem"
    ),
    (
        "messaging_calling_issue",
        "iMessage / messages / FaceTime / calls / notifications"
    ),
    (
        "other_unclear",
        "does not clearly belong to another intent"
    ),
]


def load_rows():
    with open(INPUT_PATH, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_rows(rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

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
        writer.writerows(rows)


def print_separator():
    print("\n" + "=" * 80)


def show_intents():
    print("\nChoose an intent:\n")

    for index, (name, description) in enumerate(INTENTS, start=1):
        print(f"{index:2}. {name}")
        print(f"    {description}")

    print("\nCommands:")
    print("  s = skip this example")
    print("  q = save and quit")


def get_intent():

    while True:

        show_intents()

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            return "QUIT"

        if choice == "s":
            return "SKIP"

        if choice.isdigit():

            number = int(choice)

            if 1 <= number <= len(INTENTS):
                return INTENTS[number - 1][0]

        print("\n❌ Invalid choice. Please enter a valid number.")


def label_examples(rows):

    total = len(rows)

    labeled_count = sum(
        1
        for row in rows
        if row.get("intent", "").strip()
    )

    print_separator()
    print("APPLE SUPPORT — GOLDEN SET LABELING")
    print_separator()

    print(f"Total examples: {total}")
    print(f"Already labeled: {labeled_count}")
    print(f"Remaining: {total - labeled_count}")

    print("\nStart labeling...")
    print("Your progress is saved after every example.")

    for row in rows:

        # Don't label already completed examples
        if row.get("intent", "").strip():
            continue

        print_separator()

        print(f"Example: {row['example_id']}")
        print(f"Conversation: {row['conversation_id']}")

        print_separator()

        context = row.get("context", "").strip()

        if context:
            print("PREVIOUS CONTEXT:")
            print(context)
        else:
            print("PREVIOUS CONTEXT:")
            print("(none)")

        print_separator()

        print("CUSTOMER MESSAGE:")
        print(row["customer_message"])

        print_separator()

        print("HISTORICAL APPLESUPPORT REPLY:")
        print(row["historical_agent_reply"])

        print_separator()

        intent = get_intent()

        if intent == "QUIT":
            save_rows(rows)

            print_separator()
            print("Progress saved.")
            print("Exiting labeling tool.")
            print_separator()

            return

        if intent == "SKIP":
            print("\n⏭️ Example skipped.")
            continue

        row["intent"] = intent

        # Escalation and notes will be labeled later.
        save_rows(rows)

        labeled_count += 1

        print(f"\n✅ Labeled as: {intent}")
        print(f"Progress: {labeled_count}/{total}")

    save_rows(rows)

    print_separator()
    print("🎉 GOLDEN SET LABELING COMPLETE")
    print_separator()

    print(f"Total examples: {total}")
    print(f"Labeled examples: {labeled_count}")
    print(f"Saved to: {OUTPUT_PATH}")


def main():

    if not INPUT_PATH.exists():
        print(f"❌ File not found: {INPUT_PATH}")
        return

    rows = load_rows()

    if not rows:
        print("❌ golden_sample.csv is empty.")
        return

    label_examples(rows)


if __name__ == "__main__":
    main()