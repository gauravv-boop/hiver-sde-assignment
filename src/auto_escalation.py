import pandas as pd
import re

INPUT = "data/golden_sample.csv"
OUTPUT = "data/escalation_suggestions.csv"

df = pd.read_csv(INPUT)

def suggest(row):
    text = (
        str(row.get("customer_message", "")) + " " +
        str(row.get("context", "")) + " " +
        str(row.get("historical_agent_reply", ""))
    ).lower()

    # Strong escalation signals
    strong_yes = [
        "nothing has helped",
        "still not working",
        "still doesn't work",
        "still does not work",
        "issue continues",
        "continues to",
        "need additional support",
        "additional support",
        "speak to someone",
        "human support",
        "talk to someone",
        "can't fix",
        "cannot fix",
        "tried everything",
        "already tried",
        "after trying",
        "send us a dm",
        "join us in dm",
        "continue in dm"
    ]

    # Resolved / acknowledgement signals
    resolved = [
        "all fixed",
        "fixed now",
        "issue resolved",
        "problem solved",
        "thanks for the help",
        "thank you for the help",
        "will do",
        "thanks!"
    ]

    if any(x in text for x in strong_yes):
        return "YES"

    if any(x in text for x in resolved):
        return "NO"

    # Serious account/security/payment cases
    sensitive = [
        "hacked",
        "stolen",
        "fraud",
        "unauthorized",
        "account compromised",
        "someone accessed",
        "unknown charge",
        "credit card stolen"
    ]

    if any(x in text for x in sensitive):
        return "YES"

    # Default: normal troubleshooting can be handled by AI
    return "NO"


# Only generate suggestions for currently blank escalation labels
df["suggested_escalation"] = df.apply(suggest, axis=1)

# Keep existing human labels untouched
df["final_escalation"] = df["escalation"]

df.loc[
    df["final_escalation"].isna() |
    (df["final_escalation"].astype(str).str.strip() == ""),
    "final_escalation"
] = df["suggested_escalation"]

df.to_csv(OUTPUT, index=False)

print("Done!")
print(f"Total examples: {len(df)}")
print("Suggested escalation distribution:")
print(df["suggested_escalation"].value_counts())
print(f"\nSaved to: {OUTPUT}")