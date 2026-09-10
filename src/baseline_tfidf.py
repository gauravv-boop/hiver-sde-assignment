import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


INPUT_FILE = "data/golden_sample.csv"


# Load golden evaluation set
df = pd.read_csv(INPUT_FILE)

# Only use examples having a manually assigned intent
df = df.dropna(subset=["intent"])

df["customer_message"] = df["customer_message"].fillna("")

X = df["customer_message"].astype(str)
y = df["intent"].astype(str)


# TF-IDF + Logistic Regression
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# Train
model.fit(X, y)

# Predict
predictions = model.predict(X)


print("======================================")
print("BASELINE 2 - TF-IDF + LOGISTIC REGRESSION")
print("======================================")

print("\nEvaluated examples:", len(df))

accuracy = accuracy_score(y, predictions)

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)

labels = sorted(y.unique())

print("\nLabels:")
print(labels)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y,
    predictions,
    labels=labels
)

print(cm)


# Save predictions
df["tfidf_prediction"] = predictions

df.to_csv(
    "data/tfidf_results.csv",
    index=False
)

print("\nSaved predictions to: data/tfidf_results.csv")