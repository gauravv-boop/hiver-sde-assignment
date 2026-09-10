import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Apple Support AI Agent",
    page_icon="🍎",
    layout="wide"
)

INTENTS = [
    "account_login_verification",
    "app_store_app_issue",
    "ios_software_issue",
    "battery_charging_issue",
    "wifi_network_issue",
    "icloud_backup_restore",
    "billing_payment_purchase",
    "subscription_media_issue",
    "device_hardware_issue",
    "messaging_calling_issue",
    "other_unclear"
]

ESCALATION_KEYWORDS = [
    "refund",
    "chargeback",
    "hacked",
    "fraud",
    "stolen",
    "lawsuit",
    "legal",
    "police",
    "unauthorized payment",
    "account compromised",
    "security breach"
]


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_data():
    files = [
        "data/golden_sample.csv",
        "data/golden_eval.csv",
        "data/apple_support.csv"
    ]

    for file in files:
        try:
            df = pd.read_csv(file)

            if "customer_message" in df.columns:
                df = df.rename(columns={"customer_message": "text"})

            if "text" in df.columns:
                df = df[df["text"].notna()].copy()

                if "historical_agent_reply" not in df.columns:
                    if "reply" in df.columns:
                        df["historical_agent_reply"] = df["reply"]
                    else:
                        df["historical_agent_reply"] = ""

                if "intent" not in df.columns:
                    df["intent"] = "other_unclear"

                return df

        except Exception:
            continue

    return pd.DataFrame()


@st.cache_resource
def build_embeddings(texts):
    model = load_model()
    return model.encode(
        texts.tolist(),
        normalize_embeddings=True,
        show_progress_bar=False
    )


def predict_intent(results):
    valid = results[results["intent"].isin(INTENTS)]

    if len(valid) == 0:
        return "other_unclear"

    counts = valid["intent"].value_counts()

    return counts.index[0]


def escalation_decision(message):
    text = message.lower()

    matched = [
        keyword
        for keyword in ESCALATION_KEYWORDS
        if keyword in text
    ]

    if matched:
        return "Escalate", "Sensitive issue detected: " + ", ".join(matched)

    return "Auto-handle", "No high-risk escalation signal detected"


def retrieve_cases(message, df, embeddings, model, top_k=3):
    query_embedding = model.encode(
        [message],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    indices = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[indices].copy()
    results["similarity"] = scores[indices]

    return results


st.title("🍎 Apple Support AI Agent")

st.write(
    "AI-powered customer support assistant using historical AppleSupport conversations."
)

st.divider()

message = st.text_area(
    "Customer Message",
    placeholder="Example: My iPhone battery is draining very quickly after the latest update...",
    height=130
)

analyze = st.button(
    "Analyze Customer Message",
    type="primary",
    use_container_width=True
)

if analyze:

    if not message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Analyzing customer message..."):

        df = load_data()

        if df.empty:
            st.error("Support dataset could not be loaded.")
            st.stop()

        model = load_model()

        texts = df["text"].astype(str)

        embeddings = build_embeddings(texts)

        results = retrieve_cases(
            message,
            df,
            embeddings,
            model,
            top_k=3
        )

        intent = predict_intent(results)

        decision, reason = escalation_decision(message)

        best_case = results.iloc[0]

        reply = str(best_case.get("historical_agent_reply", "")).strip()

        if not reply:
            reply = (
                "Thanks for reaching out. I’m sorry you’re experiencing this issue. "
                "Please share a few more details so we can help troubleshoot it."
            )

    st.success("Analysis complete")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Intent")
        st.info(intent)

    with col2:
        st.subheader("Handling Decision")

        if decision == "Escalate":
            st.error(decision)
        else:
            st.success(decision)

    st.subheader("Reason")
    st.write(reason)

    st.subheader("Draft Reply")

    st.text_area(
        "Historical-support-grounded draft",
        value=reply,
        height=180
    )

    st.subheader("Similar Historical Cases")

    display_columns = [
        "text",
        "historical_agent_reply",
        "intent",
        "similarity"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results.columns
    ]

    st.dataframe(
        results[available_columns],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Retrieval Confidence")

    similarity = float(results.iloc[0]["similarity"])

    st.progress(
        min(max(similarity, 0.0), 1.0)
    )

    st.write(
        f"Top historical case similarity: **{similarity:.3f}**"
    )

st.divider()

st.caption(
    "Hiver SDE Intern Take-Home Assignment | AppleSupport AI Support Agent"
)