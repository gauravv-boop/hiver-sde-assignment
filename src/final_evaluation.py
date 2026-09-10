import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = "data/golden_sample.csv"
OUTPUT_FILE = "data/final_evaluation_results.csv"

df = pd.read_csv(INPUT_FILE)

df["customer_message"] = df["customer_message"].fillna("").astype(str)
df["intent"] = df["intent"].fillna("other_unclear").astype(str)
df["escalation"] = df["escalation"].fillna("auto_handle").astype(str)

df = df.reset_index(drop=True)

X = df["customer_message"]
y = df["intent"]

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

tfidf_predictions = np.empty(len(df), dtype=object)
retrieval_predictions = np.empty(len(df), dtype=object)

print("========================================")
print("PROPER 5-FOLD EVALUATION")
print("========================================")
print(f"Golden examples: {len(df)}")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/5")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]

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

    model.fit(X_train, y_train)

    tfidf_predictions[test_idx] = model.predict(X_test)

print("\nLoading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(
    X.tolist(),
    show_progress_bar=True
)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    train_embeddings = embeddings[train_idx]
    train_labels = y.iloc[train_idx].tolist()

    test_embeddings = embeddings[test_idx]

    similarities = cosine_similarity(
        test_embeddings,
        train_embeddings
    )

    for row, test_index in enumerate(test_idx):
        top_indices = np.argsort(
            similarities[row]
        )[::-1][:3]

        intents = [
            train_labels[index]
            for index in top_indices
        ]

        valid_intents = [
            intent
            for intent in intents
            if intent != "other_unclear"
        ]

        if valid_intents:
            prediction = max(
                set(valid_intents),
                key=valid_intents.count
            )
        else:
            prediction = "other_unclear"

        retrieval_predictions[test_index] = prediction

df["tfidf_prediction"] = tfidf_predictions
df["retrieval_prediction"] = retrieval_predictions

tfidf_accuracy = accuracy_score(
    df["intent"],
    df["tfidf_prediction"]
)

retrieval_accuracy = accuracy_score(
    df["intent"],
    df["retrieval_prediction"]
)

print("\n========================================")
print("TF-IDF + LOGISTIC REGRESSION")
print("========================================")
print(f"5-Fold Accuracy: {tfidf_accuracy:.4f}")
print("\nClassification Report:")
print(
    classification_report(
        df["intent"],
        df["tfidf_prediction"],
        zero_division=0
    )
)

print("\n========================================")
print("SEMANTIC RETRIEVAL")
print("========================================")
print(f"5-Fold Intent Accuracy: {retrieval_accuracy:.4f}")
print("\nClassification Report:")
print(
    classification_report(
        df["intent"],
        df["retrieval_prediction"],
        zero_division=0
    )
)

escalation_keywords = [
    "refund",
    "charged twice",
    "fraud",
    "stolen",
    "hacked",
    "account compromised",
    "cannot access",
    "locked out",
    "legal",
    "lawsuit",
    "data loss",
    "permanently deleted"
]

escalation_predictions = []

for text in X:
    text = text.lower()

    prediction = "auto_handle"

    for keyword in escalation_keywords:
        if keyword in text:
            prediction = "escalate"
            break

    escalation_predictions.append(prediction)

df["escalation_prediction"] = escalation_predictions

escalation_accuracy = accuracy_score(
    df["escalation"],
    df["escalation_prediction"]
)

print("\n========================================")
print("ESCALATION")
print("========================================")
print(f"Accuracy: {escalation_accuracy:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        df["escalation"],
        df["escalation_prediction"]
    )
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("SAVED")
print("========================================")
print(OUTPUT_FILE)
print("\nEvaluation complete.")