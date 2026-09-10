# 🍎 Apple Support AI Agent

> AI-powered customer support agent built using historical AppleSupport conversations from the Customer Support on Twitter dataset.

**Author:** Gaurav Kumar

**Live Demo:** https://hiver-sde-assignment-2nkqptrz6ylwbbduekzrvd.streamlit.app/

---

## 1. Project Overview

This project builds an AI-assisted customer support agent for AppleSupport using historical customer support conversations from the Customer Support on Twitter dataset.

Given a new customer message, the system:

1. Classifies the message into a predefined support intent.
2. Retrieves historically similar AppleSupport conversations.
3. Generates a support draft grounded in historical responses.
4. Decides whether the issue can be auto-handled or should be escalated.
5. Shows the historical cases used as supporting evidence.

The main objective is to build a support workflow that is grounded in how similar issues were handled historically rather than producing completely generic chatbot responses.

---

## 2. Problem Framing

Customer support conversations contain recurring problem patterns.

For AppleSupport, common categories include:

- iOS and software issues
- App Store and application problems
- Battery and charging issues
- Wi-Fi and network issues
- Account and login problems
- iCloud, backup and restore problems
- Billing and purchase issues
- Subscription and media issues
- Device hardware issues
- Messaging and calling issues

The system follows this workflow:

```text
                Customer Message
                       |
                       v
              Intent Classification
                       |
                       v
            Historical Retrieval
                       |
             +---------+---------+
             |                   |
             v                   v
       Draft Reply        Handling Decision
                               |
                     +---------+---------+
                     |                   |
                     v                   v
                 Auto-handle          Escalate
