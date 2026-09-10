import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/golden_eval.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3


# ============================================================
# LOAD DATA
# ============================================================

print("Loading golden evaluation data...")

df = pd.read_csv(INPUT_FILE)

df["customer_message"] = df["customer_message"].fillna("").astype(str)
df["historical_agent_reply"] = (
    df["historical_agent_reply"]
    .fillna("")
    .astype(str)
)

print("Loaded examples:", len(df))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# ============================================================
# CREATE HISTORICAL CORPUS
# ============================================================

documents = df["customer_message"].tolist()

print("\nCreating embeddings...")

embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Make absolutely sure embeddings are 2D
embeddings = np.asarray(embeddings).reshape(len(documents), -1)

print("Embedding shape:", embeddings.shape)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve(query, top_k=TOP_K):

    # Create embedding for incoming customer message
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # IMPORTANT:
    # cosine_similarity expects 2D arrays
    query_embedding = np.asarray(query_embedding).reshape(1, -1)

    # Calculate similarity
    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Get highest scoring examples
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "customer_message": df.iloc[index]["customer_message"],
            "historical_agent_reply": df.iloc[index]["historical_agent_reply"],
            "intent": df.iloc[index].get("intent", ""),
            "similarity": float(scores[index])
        })

    return results


# ============================================================
# INTERACTIVE AGENT
# ============================================================

print("\n======================================")
print("APPLE SUPPORT RETRIEVAL AGENT")
print("======================================")

print("Type a customer message.")
print("Type 'exit' to stop.\n")


while True:

    query = input("Enter customer message: ").strip()

    if query.lower() == "exit":
        print("\nExiting...")
        break

    if not query:
        print("Please enter a message.\n")
        continue

    # Retrieve similar historical cases
    results = retrieve(query)

    print("\n--------------------------------------")
    print("TOP HISTORICAL CASES")
    print("--------------------------------------")

    for i, result in enumerate(results, start=1):

        print(f"\n### Result {i}")

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Intent: "
            f"{result['intent']}"
        )

        print(
            "\nCustomer message:"
        )

        print(
            result["customer_message"]
        )

        print(
            "\nHistorical AppleSupport reply:"
        )

        print(
            result["historical_agent_reply"]
        )

    print("\n======================================\n")