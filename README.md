# Apple Support AI Agent

AI-powered customer support agent built for the Hiver SDE Intern Take-Home Assignment using historical AppleSupport conversations from the Customer Support on Twitter dataset.

The system classifies incoming customer messages, retrieves historically similar support cases, drafts a historically grounded response, and decides whether the case should be auto-handled or escalated.

## Live Demo

The application is deployed as a public Streamlit app.

Live demo: https://hiver-sde-assignment-2nkqptrz6ylwbbduekzrvd.streamlit.app

## Problem Framing

Customer support conversations are often repetitive, but the correct response depends on the type of issue and how the brand historically handled similar problems.

For this project, AppleSupport was selected as the target brand.

A good support agent should:

- Identify the customer's primary intent.
- Find historically similar customer issues.
- Use historical support responses as grounding evidence.
- Produce a concise support-oriented draft.
- Avoid automatically handling cases with high-risk escalation signals.
- Provide a reason for the handling decision.

### What I chose not to build

I intentionally did not build:

- A fully autonomous production support system.
- Direct access to customer accounts or private Apple systems.
- Automated actions such as refunds, password resets, or account changes.
- A large-scale generative model fine-tuning pipeline.
- Full-dataset inference during evaluation.

The focus was on building an explainable and reproducible support-agent prototype using historical support data.

---

## Dataset

The project uses the Kaggle Customer Support on Twitter dataset from the `thoughtvector/customer-support-on-twitter` dataset.

The full dataset contains approximately 2.8 million tweets.

Relevant columns include:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

`inbound=True` is treated as a customer message and `inbound=False` as a brand/support response.

AppleSupport was selected as the target brand.

The extracted AppleSupport data contains approximately 106K AppleSupport tweets.

Conversation relationships were reconstructed using tweet and response IDs to preserve multi-turn support conversations.

The large raw dataset is intentionally not committed to GitHub.

---

## Intent Taxonomy

A compact 11-class intent taxonomy was created after inspecting AppleSupport customer conversations.

| Intent | Description |
|---|---|
| `account_login_verification` | Login, account access, verification and authentication issues |
| `app_store_app_issue` | App Store or application-related problems |
| `ios_software_issue` | iOS, updates and general software problems |
| `battery_charging_issue` | Battery drain, charging and power issues |
| `wifi_network_issue` | Wi-Fi and network connectivity issues |
| `icloud_backup_restore` | iCloud, backup and restore issues |
| `billing_payment_purchase` | Billing, payment and purchase problems |
| `subscription_media_issue` | Subscriptions and media services |
| `device_hardware_issue` | Physical device or hardware-related problems |
| `messaging_calling_issue` | Messages, calls and communication problems |
| `other_unclear` | Cases that do not fit the defined support categories clearly |

The taxonomy intentionally keeps the number of classes small enough for a practical support-routing system.

---

## Golden Evaluation Set

A 200-example golden evaluation set was created.

Each example contains:

- Customer message
- Conversation context
- Historical agent reply
- Intent label
- Escalation label/reason
- Notes

The examples were sampled from the AppleSupport conversation data and manually reviewed for intent assignment.

The final intent distribution contains all 11 taxonomy classes, with the largest class being `ios_software_issue` and `other_unclear`.

Escalation labels were assisted using a rule-based escalation process and subsequently applied to the golden set. Therefore, the escalation labels should not be interpreted as fully independent human-only ground truth.

The golden set is kept separate from the large historical corpus for evaluation purposes.

---

## System Architecture

```text
Customer Message
       |
       v
Intent Classification
       |
       +--------------------+
       |                    |
       v                    v
Historical Retrieval    Escalation Rules
       |                    |
       v                    v
Similar AppleSupport    Auto-handle /
Cases                   Escalate
       |
       v
Historical Support
Response Grounding
       |
       v
Draft Reply
