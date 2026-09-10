import pandas as pd
from pathlib import Path

golden = Path("data/golden_sample.csv")
suggestions = Path("data/escalation_suggestions.csv")
backup = Path("data/golden_sample_backup.csv")
review = Path("data/escalation_review.csv")

df = pd.read_csv(golden)
sug = pd.read_csv(suggestions)

# Backup original before changing anything
if not backup.exists():
    df.to_csv(backup, index=False)

# Detect suggestion column
possible = ["suggested_escalation", "escalation", "suggestion"]
col = next((c for c in possible if c in sug.columns), None)
if col is None:
    raise ValueError(f"Could not find escalation suggestion column. Columns: {list(sug.columns)}")

# Map by example_id
mapping = sug.set_index("example_id")[col].astype(str).str.upper().to_dict()

df["escalation"] = df["example_id"].map(mapping).map({
    "YES": "escalate",
    "NO": "auto_handle"
})

# Preserve a useful reason for the automatically generated label
df["escalation_reason"] = df["escalation"].map({
    "escalate": "Rule-based escalation suggestion; human review recommended.",
    "auto_handle": "Rule-based auto-handle suggestion."
})

df.to_csv(golden, index=False)

# Create the smaller human-review file for YES cases
review_df = df[df["escalation"] == "escalate"].copy()
review_df.to_csv(review, index=False)

print("DONE")
print("Total examples:", len(df))
print("Escalation labeled:", df["escalation"].notna().sum())
print("\nDistribution:")
print(df["escalation"].value_counts(dropna=False))
print("\nBackup:", backup)
print("Review file:", review)
