# 🍎 Apple Support AI Agent

### AI-assisted customer support agent grounded in historical AppleSupport conversations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Prototype-success)]()

**Live Demo:** https://hiver-sde-assignment-2nkqptrz6ylwbbduekzrvd.streamlit.app

---

## 🎯 What This Project Does

Customer-support teams receive large volumes of repetitive requests. A significant amount of the knowledge required to answer these requests already exists in historical support conversations.

This project explores how that historical knowledge can be turned into a practical AI-assisted support workflow.

Given a new customer message, the system:

> **Classifies the issue → Finds similar historical cases → Grounds the response → Decides whether to handle or escalate**

The goal is not to build a fully autonomous support bot, but a **transparent support assistant** that helps produce consistent, evidence-backed responses while identifying cases that may require human intervention.

---

## ✨ Key Capabilities

### 1. Intent Classification

Incoming customer messages are mapped to a compact support taxonomy containing 11 intents:

| Intent | Description |
|---|---|
| `account_login_verification` | Account, login and verification issues |
| `app_store_app_issue` | App Store and application problems |
| `ios_software_issue` | iOS, updates and software issues |
| `battery_charging_issue` | Battery drain and charging problems |
| `wifi_network_issue` | Wi-Fi and network connectivity |
| `icloud_backup_restore` | iCloud, backup and restore |
| `billing_payment_purchase` | Billing, payment and purchase issues |
| `subscription_media_issue` | Subscription and media services |
| `device_hardware_issue` | Hardware and device problems |
| `messaging_calling_issue` | Messages, calls and communication |
| `other_unclear` | Ambiguous or unsupported requests |

---

### 2. 🔎 Historical Case Retrieval

The system searches historical AppleSupport conversations to find cases that are semantically similar to the incoming request.

Sentence embeddings are generated using:

**`all-MiniLM-L6-v2`**

Similarity is calculated using cosine similarity.

The UI exposes the retrieved historical cases so that the response is not treated as a black box.

---

### 3. 💬 Grounded Reply Drafting

Retrieved historical support responses are used as grounding evidence for the response draft.

This provides two advantages:

- Responses remain aligned with previously observed support behaviour.
- The agent can show the historical cases that influenced the draft.

The system therefore prioritizes **grounded assistance over unrestricted response generation**.

---

### 4. 🛡️ Handling Decision

Each request receives a handling recommendation:

**Auto-handle**

or

**Escalate**

The system also provides a reason for the decision.

This is designed as a safety layer so that uncertain or high-risk requests can be routed to a human rather than being blindly automated.

---

# 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │   Customer Message  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Intent Classifier  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    │  Historical Cases   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Historical Evidence │
                    │ + Support Responses │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
       ┌──────────────────┐       ┌──────────────────┐
       │  Reply Drafting  │       │ Escalation Check │
       └────────┬─────────┘       └────────┬─────────┘
                │                          │
                └────────────┬─────────────┘
                             ▼
                    ┌─────────────────────┐
                    │    Final Result     │
                    │ Intent + Reply +    │
                    │ Evidence + Decision │
                    └─────────────────────┘
