import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os


FILE = "data/golden_sample.csv"

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


class GoldenSetLabeler:

    def __init__(self, root):
        self.root = root
        self.root.title("AppleSupport Golden Set Labeler")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 700)

        self.df = pd.read_csv(FILE)

        # Make sure required columns exist
        if "intent" not in self.df.columns:
            self.df["intent"] = ""

        if "escalation" not in self.df.columns:
            self.df["escalation"] = ""

        if "notes" not in self.df.columns:
            self.df["notes"] = ""

        self.df["intent"] = self.df["intent"].fillna("").astype(str)

        self.index = None

        self.build_ui()
        self.load_next()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def build_ui(self):

        # Header
        header = tk.Frame(self.root)
        header.pack(fill="x", padx=20, pady=(15, 5))

        tk.Label(
            header,
            text="APPLE SUPPORT — GOLDEN SET LABELING",
            font=("Arial", 22, "bold")
        ).pack()

        self.counter = tk.Label(
            header,
            text="",
            font=("Arial", 13)
        )
        self.counter.pack(pady=5)

        # Main content
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=20)

        # Previous context
        tk.Label(
            main,
            text="PREVIOUS CONTEXT",
            font=("Arial", 11, "bold"),
            anchor="w"
        ).pack(fill="x", pady=(5, 2))

        self.context = tk.Text(
            main,
            height=5,
            wrap="word",
            font=("Arial", 11)
        )
        self.context.pack(fill="x", pady=(0, 8))
        self.context.config(state="disabled")

        # Customer message
        tk.Label(
            main,
            text="CUSTOMER MESSAGE",
            font=("Arial", 11, "bold"),
            anchor="w"
        ).pack(fill="x", pady=(3, 2))

        self.customer = tk.Text(
            main,
            height=4,
            wrap="word",
            font=("Arial", 11)
        )
        self.customer.pack(fill="x", pady=(0, 8))
        self.customer.config(state="disabled")

        # Historical reply
        tk.Label(
            main,
            text="HISTORICAL APPLESUPPORT REPLY",
            font=("Arial", 11, "bold"),
            anchor="w"
        ).pack(fill="x", pady=(3, 2))

        self.reply = tk.Text(
            main,
            height=4,
            wrap="word",
            font=("Arial", 11)
        )
        self.reply.pack(fill="x", pady=(0, 8))
        self.reply.config(state="disabled")

        # Intent title
        tk.Label(
            main,
            text="SELECT INTENT",
            font=("Arial", 12, "bold"),
            anchor="w"
        ).pack(fill="x", pady=(3, 5))

        # Intent buttons
        self.button_frame = tk.Frame(main)
        self.button_frame.pack(fill="x")

        for i, intent in enumerate(INTENTS):

            row = i // 2
            col = i % 2

            button = tk.Button(
                self.button_frame,
                text=f"{i + 1}. {intent}",
                font=("Arial", 10, "bold"),
                anchor="w",
                height=2,
                command=lambda x=intent: self.save_intent(x)
            )

            button.grid(
                row=row,
                column=col,
                sticky="ew",
                padx=5,
                pady=3
            )

        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)

        # Skip button
        self.skip_button = tk.Button(
            main,
            text="SKIP",
            font=("Arial", 11, "bold"),
            height=2,
            command=self.skip_example
        )
        self.skip_button.pack(fill="x", padx=5, pady=(8, 5))

        # Footer
        self.status = tk.Label(
            main,
            text="",
            font=("Arial", 10)
        )
        self.status.pack(pady=3)

    # ---------------------------------------------------------
    # Helper to display text
    # ---------------------------------------------------------

    def set_text(self, widget, text):

        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", str(text))
        widget.config(state="disabled")

    # ---------------------------------------------------------
    # Find next unlabeled example
    # ---------------------------------------------------------

    def load_next(self):

        unlabeled = self.df[
            self.df["intent"].str.strip() == ""
        ]

        if len(unlabeled) == 0:

            self.counter.config(
                text=f"Completed: {len(self.df)} / {len(self.df)}"
            )

            self.set_text(
                self.context,
                "All 200 examples have been labeled."
            )

            self.set_text(
                self.customer,
                "GOLDEN SET COMPLETE"
            )

            self.set_text(
                self.reply,
                "You can now run the evaluation."
            )

            for child in self.button_frame.winfo_children():
                child.config(state="disabled")

            self.skip_button.config(state="disabled")

            messagebox.showinfo(
                "Complete",
                "Intent labeling complete!\n\n200/200 examples labeled."
            )

            return

        # First unlabeled row
        self.index = unlabeled.index[0]

        row_number = self.index + 1
        labeled_count = len(self.df) - len(unlabeled)

        self.counter.config(
            text=f"Example {row_number} / {len(self.df)}   |   "
                 f"Labeled: {labeled_count} / {len(self.df)}"
        )

        context = self.df.loc[self.index, "context"]
        customer = self.df.loc[self.index, "customer_message"]
        reply = self.df.loc[self.index, "historical_agent_reply"]

        if pd.isna(context):
            context = "(No previous context)"

        if pd.isna(customer):
            customer = ""

        if pd.isna(reply):
            reply = ""

        self.set_text(self.context, context)
        self.set_text(self.customer, customer)
        self.set_text(self.reply, reply)

        self.status.config(
            text="Choose the best matching intent."
        )

    # ---------------------------------------------------------
    # Save intent
    # ---------------------------------------------------------

    def save_intent(self, intent):

        if self.index is None:
            return

        self.df.at[self.index, "intent"] = intent

        self.df.to_csv(
            FILE,
            index=False
        )

        self.status.config(
            text=f"Saved: {intent}"
        )

        self.load_next()

    # ---------------------------------------------------------
    # Skip
    # ---------------------------------------------------------

    def skip_example(self):

        if self.index is None:
            return

        # Move this row temporarily to the bottom by
        # storing its index in a skip list.
        current = self.index

        # Use a separate attribute for skipped examples
        if not hasattr(self, "skipped"):
            self.skipped = set()

        self.skipped.add(current)

        unlabeled = [
            i for i in self.df.index
            if self.df.at[i, "intent"].strip() == ""
            and i not in self.skipped
        ]

        if not unlabeled:

            self.status.config(
                text="No more unskipped examples. Review skipped examples."
            )

            # Reset skips so they can be reviewed
            self.skipped.clear()

            self.load_next()
            return

        self.index = unlabeled[0]

        row_number = self.index + 1

        self.counter.config(
            text=f"Example {row_number} / {len(self.df)}   |   "
                 f"Labeled: {len(self.df) - len(unlabeled)} / {len(self.df)}"
        )

        context = self.df.loc[self.index, "context"]
        customer = self.df.loc[self.index, "customer_message"]
        reply = self.df.loc[self.index, "historical_agent_reply"]

        self.set_text(
            self.context,
            "(No previous context)" if pd.isna(context) else context
        )

        self.set_text(
            self.customer,
            "" if pd.isna(customer) else customer
        )

        self.set_text(
            self.reply,
            "" if pd.isna(reply) else reply
        )

        self.status.config(
            text="Skipped. This example will be reviewed later."
        )


# ---------------------------------------------------------
# Run application
# ---------------------------------------------------------

if __name__ == "__main__":

    if not os.path.exists(FILE):
        print(f"ERROR: File not found: {FILE}")
        print("Make sure you are running this from the project root.")
        raise SystemExit(1)

    root = tk.Tk()

    app = GoldenSetLabeler(root)

    root.mainloop()