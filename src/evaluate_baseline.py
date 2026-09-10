import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT_FILE = "data/baseline_keyword_results.csv"

df = pd.read_csv(INPUT_FILE)

# Remove rows where either true or predicted intent is missing
df = df.dropna(subset=["intent", "predicted_intent"])

y_true = df["intent"].astype(str)
y_pred = df["predicted_intent"].astype(str)

print("===================================")
print("BASELINE 1 - KEYWORD/RULE MODEL")
print("===================================")

print("\nEvaluated examples:", len(df))

print("\nAccuracy:")
print(round(accuracy_score(y_true, y_pred), 4))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
labels = sorted(y_true.unique())

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("\nLabels:")
print(labels)

print("\nMatrix:")
print(cm)