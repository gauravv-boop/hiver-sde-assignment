import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

INPUT_FILE = "data/final_evaluation_results.csv"
OUTPUT_FILE = "data/reply_judge_results.csv"
SAMPLE_SIZE = 30
MODEL_NAME = "google/flan-t5-base"

df = pd.read_csv(INPUT_FILE)

df["customer_message"] = df["customer_message"].fillna("").astype(str)

if "retrieved_reply" in df.columns:
    reply_column = "retrieved_reply"
elif "reply" in df.columns:
    reply_column = "reply"
elif "historical_agent_reply" in df.columns:
    reply_column = "historical_agent_reply"
else:
    raise ValueError("No reply column found")

df[reply_column] = df[reply_column].fillna("").astype(str)

df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).reset_index(drop=True)

print("Loading local judge model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Judge device: {device}")
print(f"Examples to judge: {len(df)}")


def get_score(customer, reply, dimension):
    prompts = {
        "correctness": f"""
Rate the correctness of this Apple support reply from 1 to 5.

Customer:
{customer}

Reply:
{reply}

1 = incorrect
2 = mostly incorrect
3 = partially correct
4 = mostly correct
5 = fully correct

Return only one number.
""",
        "groundedness": f"""
Rate how well this Apple support reply is grounded in the available support response from 1 to 5.

Customer:
{customer}

Reply:
{reply}

1 = unsupported
2 = weakly grounded
3 = partially grounded
4 = well grounded
5 = strongly grounded

Return only one number.
""",
        "relevance": f"""
Rate how relevant this Apple support reply is to the customer issue from 1 to 5.

Customer:
{customer}

Reply:
{reply}

1 = irrelevant
2 = weakly relevant
3 = somewhat relevant
4 = highly relevant
5 = directly relevant

Return only one number.
""",
        "safety": f"""
Rate the safety of this Apple support reply from 1 to 5.

Customer:
{customer}

Reply:
{reply}

1 = unsafe
2 = potentially risky
3 = acceptable
4 = safe
5 = very safe and avoids unsupported claims

Return only one number.
""",
        "overall": f"""
Rate the overall quality of this Apple support reply from 1 to 5.

Customer:
{customer}

Reply:
{reply}

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Return only one number.
"""
    }

    inputs = tokenizer(
        prompts[dimension],
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        num_beams=4
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    match = re.search(r"\b([1-5])\b", text)

    if match:
        return int(match.group(1))

    return np.nan


results = []

for i, row in df.iterrows():
    print(f"Judging {i + 1}/{len(df)}")

    customer = row["customer_message"]
    reply = row[reply_column]

    correctness = get_score(
        customer,
        reply,
        "correctness"
    )

    groundedness = get_score(
        customer,
        reply,
        "groundedness"
    )

    relevance = get_score(
        customer,
        reply,
        "relevance"
    )

    safety = get_score(
        customer,
        reply,
        "safety"
    )

    overall = get_score(
        customer,
        reply,
        "overall"
    )

    results.append({
        "example_id": row.get("example_id", i + 1),
        "customer_message": customer,
        "reply": reply,
        "correctness": correctness,
        "groundedness": groundedness,
        "relevance": relevance,
        "safety": safety,
        "overall": overall
    })


result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("LLM-AS-JUDGE RESULTS")
print("========================================")

print(f"Average correctness: {result_df['correctness'].mean():.2f}")
print(f"Average groundedness: {result_df['groundedness'].mean():.2f}")
print(f"Average relevance: {result_df['relevance'].mean():.2f}")
print(f"Average safety: {result_df['safety'].mean():.2f}")
print(f"Average overall: {result_df['overall'].mean():.2f}")

print("\nValid scores:")
print(
    result_df[
        [
            "correctness",
            "groundedness",
            "relevance",
            "safety",
            "overall"
        ]
    ].notna().sum()
)

print("\nSaved:")
print(OUTPUT_FILE)