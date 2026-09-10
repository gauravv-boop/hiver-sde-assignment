import pandas as pd
import re
from pathlib import Path

INPUT_FILE = "data/golden_eval.csv"
OUTPUT_FILE = "data/baseline_keyword_results.csv"


INTENT_KEYWORDS = {
    "account_login_verification": [
        "login", "log in", "password", "account", "verification",
        "verify", "locked", "sign in", "signin", "apple id"
    ],

    "app_store_app_issue": [
        "app store", "appstore", "app", "application",
        "download", "install", "update app", "crash"
    ],

    "ios_software_issue": [
        "ios", "update", "software", "settings", "screen",
        "brightness", "rotation", "gesture", "keyboard"
    ],

    "battery_charging_issue": [
        "battery", "charging", "charge", "charger",
        "drain", "battery life", "power"
    ],

    "wifi_network_issue": [
        "wifi", "wi-fi", "network", "internet",
        "hotspot", "bluetooth", "signal", "cellular"
    ],

    "icloud_backup_restore": [
        "icloud", "backup", "restore", "restore from backup",
        "icloud drive"
    ],

    "billing_payment_purchase": [
        "payment", "paid", "charge", "charged", "billing",
        "purchase", "refund", "credit card", "debit card"
    ],

    "subscription_media_issue": [
        "subscription", "apple music", "music", "itunes",
        "podcast", "tv", "renewal"
    ],

    "device_hardware_issue": [
        "iphone", "ipad", "macbook", "mac",
        "screen", "display", "camera", "speaker",
        "button", "broken", "physical"
    ],

    "messaging_calling_issue": [
        "imessage", "message", "messaging", "sms",
        "call", "calling", "facetime", "text"
    ]
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def predict_intent(text):
    text = clean_text(text)

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "other_unclear"

    return best_intent


def main():
    df = pd.read_csv(INPUT_FILE)

    df["predicted_intent"] = df["customer_message"].apply(
        predict_intent
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("Baseline 1 completed!")
    print("Total examples:", len(df))
    print("\nPredicted distribution:")
    print(df["predicted_intent"].value_counts())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()