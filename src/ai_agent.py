import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = "data/golden_sample.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

ESCALATION_KEYWORDS = [
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

print("Loading historical AppleSupport data...")

df = pd.read_csv(INPUT_FILE)
df["customer_message"] = df["customer_message"].fillna("")
df["historical_agent_reply"] = df["historical_agent_reply"].fillna("")
df["intent"] = df["intent"].fillna("other_unclear")

print(f"Loaded examples: {len(df)}")

print("\nLoading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded.")

print("\nCreating historical embeddings...")
embeddings = model.encode(
    df["customer_message"].astype(str).tolist(),
    show_progress_bar=True
)

print(f"Embedding shape: {embeddings.shape}")


def retrieve(query, top_k=TOP_K):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in indices:
        results.append({
            "customer_message": df.iloc[index]["customer_message"],
            "historical_agent_reply": df.iloc[index]["historical_agent_reply"],
            "intent": df.iloc[index]["intent"],
            "similarity": float(similarities[index])
        })

    return results


def detect_intent(results):
    intents = [
        r["intent"]
        for r in results
        if r["intent"] != "other_unclear"
    ]

    if not intents:
        return "other_unclear"

    return max(set(intents), key=intents.count)


def decide_escalation(query, intent, results):
    text = query.lower()

    for keyword in ESCALATION_KEYWORDS:
        if keyword in text:
            return "escalate", f"Sensitive or high-risk issue detected: {keyword}"

    best_similarity = results[0]["similarity"]

    if best_similarity < 0.45:
        return "escalate", "Low similarity to historical support cases"

    if intent == "other_unclear":
        return "escalate", "Intent could not be confidently identified"

    return "auto_handle", "Similar historical cases provide sufficient grounding"


def generate_reply(query, intent, escalation, results):
    best = results[0]
    historical_reply = str(best["historical_agent_reply"]).strip()

    if not historical_reply:
        return "Thanks for reaching out. Please share a few more details so we can help troubleshoot this issue."

    if escalation == "escalate":
        return (
            "Thanks for reaching out. We understand this may need further assistance. "
            "Please share the relevant details with our support team so they can look into this for you."
        )

    return historical_reply


def run_agent(customer_message):
    results = retrieve(customer_message)

    intent = detect_intent(results)

    escalation, reason = decide_escalation(
        customer_message,
        intent,
        results
    )

    reply = generate_reply(
        customer_message,
        intent,
        escalation,
        results
    )

    return {
        "intent": intent,
        "escalation": escalation,
        "reason": reason,
        "reply": reply,
        "evidence": results
    }


print("\n========================================")
print("APPLE SUPPORT AI AGENT")
print("========================================")
print("\nType a customer message.")
print("Type 'exit' to stop.\n")

while True:
    customer_message = input("Customer message: ").strip()

    if customer_message.lower() == "exit":
        break

    if not customer_message:
        continue

    print("\nProcessing...\n")

    result = run_agent(customer_message)

    print("INTENT:")
    print(result["intent"])

    print("\nESCALATION:")
    print(result["escalation"])

    print("\nREASON:")
    print(result["reason"])

    print("\nDRAFT REPLY:")
    print(result["reply"])

    print("\nEVIDENCE:")
    for i, evidence in enumerate(result["evidence"], 1):
        print(f"\nResult {i}")
        print(f"Similarity: {evidence['similarity']:.4f}")
        print(f"Intent: {evidence['intent']}")
        print(f"Customer: {evidence['customer_message']}")
        print(f"Historical reply: {evidence['historical_agent_reply']}")

    print("\n" + "=" * 50 + "\n")