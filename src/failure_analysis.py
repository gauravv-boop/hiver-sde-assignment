import pandas as pd

INPUT_FILE = "data/final_evaluation_results.csv"
OUTPUT_FILE = "data/failure_analysis.txt"

df = pd.read_csv(INPUT_FILE)

print("Loading evaluation results...")

lines = []

lines.append("FAILURE ANALYSIS")
lines.append("=" * 60)
lines.append(f"Total evaluation examples: {len(df)}")
lines.append("")

if "intent" in df.columns and "tfidf_prediction" in df.columns:
    intent_failures = df[
        df["intent"].fillna("").astype(str) !=
        df["tfidf_prediction"].fillna("").astype(str)
    ]
else:
    intent_failures = pd.DataFrame()

lines.append(f"Intent classification failures: {len(intent_failures)}")
lines.append("")

if len(intent_failures) > 0:
    lines.append("TOP INTENT FAILURES")
    lines.append("-" * 60)

    for _, row in intent_failures.head(5).iterrows():
        lines.append(f"Customer: {row['customer_message']}")
        lines.append(f"Expected: {row['intent']}")
        lines.append(f"Predicted: {row['tfidf_prediction']}")
        lines.append("")

if "retrieval_prediction" in df.columns:
    retrieval_failures = df[
        df["retrieval_prediction"].fillna("").astype(str) !=
        df["intent"].fillna("").astype(str)
    ]
else:
    retrieval_failures = pd.DataFrame()

lines.append(f"Retrieval intent failures: {len(retrieval_failures)}")
lines.append("")

if len(retrieval_failures) > 0:
    lines.append("TOP RETRIEVAL FAILURES")
    lines.append("-" * 60)

    for _, row in retrieval_failures.head(5).iterrows():
        lines.append(f"Customer: {row['customer_message']}")
        lines.append(f"Expected: {row['intent']}")
        lines.append(f"Predicted: {row['retrieval_prediction']}")
        lines.append("")

if "escalation_reason" in df.columns:
    escalation_counts = df["escalation_reason"].fillna("").value_counts()
    lines.append("ESCALATION REASONS")
    lines.append("-" * 60)

    for reason, count in escalation_counts.head(10).items():
        if reason:
            lines.append(f"{reason}: {count}")

lines.append("")
lines.append("MAIN FAILURE HYPOTHESES")
lines.append("-" * 60)
lines.append("1. Ambiguous customer messages can map to multiple intents.")
lines.append("2. Short messages provide insufficient semantic context.")
lines.append("3. Keyword-heavy issues can overlap across device and software categories.")
lines.append("4. Retrieval can surface semantically similar but operationally different cases.")
lines.append("5. Historical replies may not always contain enough context to safely generate a complete answer.")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("")
print("=" * 60)
print("FAILURE ANALYSIS")
print("=" * 60)
print(f"Total examples: {len(df)}")
print(f"Intent failures: {len(intent_failures)}")
print(f"Retrieval failures: {len(retrieval_failures)}")
print("")
print("Saved:")
print(OUTPUT_FILE)