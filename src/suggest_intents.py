import pandas as pd
import re
from pathlib import Path


INPUT_FILE = Path("data/golden_sample.csv")
OUTPUT_FILE = Path("data/intent_suggestions.csv")


INTENT_KEYWORDS = {
    "account_login_verification": [
        "password", "login", "log in", "sign in", "signin",
        "apple id", "appleid", "account", "verification",
        "verify", "2fa", "two factor", "locked out",
        "forgot password", "security"
    ],

    "app_store_app_issue": [
        "app store", "appstore", "app won't", "app wont",
        "app not", "application", "download app",
        "install app", "app update", "apps", "app crashes",
        "app crash", "snapchat", "facebook", "instagram"
    ],

    "ios_software_issue": [
        "ios", "update", "updated", "updating", "upgrade",
        "downgrade", "rollback", "restore update",
        "latest update", "software", "system",
        "freeze", "freezes", "bug", "bugs", "brightness",
        "rotation", "touch", "screen", "notification",
        "background app refresh"
    ],

    "battery_charging_issue": [
        "battery", "battery drain", "battery life",
        "charging", "charge", "charger", "charging cable",
        "won't charge", "wont charge", "not charging",
        "overheating", "power"
    ],

    "wifi_network_issue": [
        "wifi", "wi-fi", "wi fi", "network",
        "internet", "hotspot", "mobile data",
        "4g", "5g", "signal", "bluetooth",
        "connection", "connect", "disconnect"
    ],

    "icloud_backup_restore": [
        "icloud", "icloud drive", "backup", "back up",
        "restore", "restoring", "backup restore",
        "sync", "synchronization", "photos backup"
    ],

    "billing_payment_purchase": [
        "payment", "paid", "pay", "charge", "charged",
        "billing", "bill", "purchase", "purchased",
        "refund", "money", "credit card", "debit card",
        "card", "invoice", "receipt", "order"
    ],

    "subscription_media_issue": [
        "subscription", "subscribed", "music",
        "apple music", "itunes", "movie", "movies",
        "tv", "apple tv", "podcast", "playlist",
        "song", "songs", "media"
    ],

    "device_hardware_issue": [
        "iphone", "ipad", "macbook", "mac book", "mac",
        "device", "phone", "screen broken", "broken",
        "cracked", "camera", "speaker", "microphone",
        "button", "home button", "hardware"
    ],

    "messaging_calling_issue": [
        "imessage", "i-message", "message", "messages",
        "text", "texts", "sms", "facetime", "face time",
        "call", "calling", "calls", "phone call"
    ],
}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Normalize punctuation
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def score_intents(text):
    """
    Score every intent using keyword matches.
    Longer/more specific phrases get slightly higher weight.
    """

    text = clean_text(text)

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            keyword = keyword.lower()

            if keyword in text:
                # Multi-word phrases are more informative
                if " " in keyword:
                    score += 3
                else:
                    score += 1

        scores[intent] = score

    return scores


def suggest_intent(customer_message, context="", historical_reply=""):
    """
    Customer message is given highest importance.
    Context/reply are used only when customer message is unclear.
    """

    customer_text = clean_text(customer_message)
    context_text = clean_text(context)
    reply_text = clean_text(historical_reply)

    customer_scores = score_intents(customer_text)

    best_intent = max(customer_scores, key=customer_scores.get)
    best_score = customer_scores[best_intent]

    # If customer message itself gives a clear signal
    if best_score >= 2:
        return best_intent

    # Use previous context when customer message is vague
    context_scores = score_intents(context_text)

    # Use historical reply as a weaker signal
    reply_scores = score_intents(reply_text)

    combined_scores = {}

    for intent in INTENT_KEYWORDS:
        combined_scores[intent] = (
            customer_scores[intent] * 5
            + context_scores[intent] * 2
            + reply_scores[intent]
        )

    best_intent = max(combined_scores, key=combined_scores.get)

    # If absolutely nothing matches, use other_unclear
    if combined_scores[best_intent] == 0:
        return "other_unclear"

    return best_intent


def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        print("Make sure data/golden_sample.csv exists.")
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded examples: {len(df)}")

    suggestions = []

    for _, row in df.iterrows():

        customer_message = row.get("customer_message", "")
        context = row.get("context", "")
        historical_reply = row.get("historical_agent_reply", "")

        suggested = suggest_intent(
            customer_message,
            context,
            historical_reply
        )

        suggestions.append(suggested)

    df["suggested_intent"] = suggestions

    # Save result
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("Done!")
    print(f"Total examples: {len(df)}")
    print()
    print("Suggested intent distribution:")
    print(df["suggested_intent"].value_counts().to_string())
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()