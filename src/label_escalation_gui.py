import csv
import tkinter as tk
from tkinter import messagebox
from pathlib import Path


INPUT_PATH = Path("data/golden_sample.csv")
OUTPUT_PATH = Path("data/golden_sample.csv")


class EscalationLabeler:

    def __init__(self, root):
        self.root = root
        self.root.title("AppleSupport Golden Set - Escalation Labeling")
        self.root.geometry("1100x800")

        self.rows = self.load_rows()

        if not self.rows:
            messagebox.showerror(
                "Error",
                "No data found in data/golden_sample.csv"
            )
            self.root.destroy()
            return

        self.current_index = 0

        # Find first unlabeled escalation
        self.find_next_unlabeled()

        self.build_ui()
        self.show_current()

    # -----------------------------
    # Load CSV
    # -----------------------------

    def load_rows(self):
        with open(
            INPUT_PATH,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            return list(csv.DictReader(file))

    # -----------------------------
    # Save CSV
    # -----------------------------

    def save_rows(self):
        if not self.rows:
            return

        fieldnames = list(self.rows[0].keys())

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(self.rows)

    # -----------------------------
    # Find next unlabeled example
    # -----------------------------

    def find_next_unlabeled(self):

        while self.current_index < len(self.rows):

            escalation = self.rows[
                self.current_index
            ].get("escalation", "").strip().lower()

            if escalation not in ["yes", "no"]:
                return

            self.current_index += 1

    # -----------------------------
    # Build UI
    # -----------------------------

    def build_ui(self):

        # Header
        header = tk.Frame(
            self.root,
            padx=15,
            pady=10
        )

        header.pack(fill="x")

        tk.Label(
            header,
            text="APPLE SUPPORT — ESCALATION LABELING",
            font=("Arial", 18, "bold")
        ).pack()

        self.progress_label = tk.Label(
            header,
            text="",
            font=("Arial", 12)
        )

        self.progress_label.pack(
            pady=(5, 0)
        )

        # Main content
        content = tk.Frame(
            self.root,
            padx=15,
            pady=5
        )

        content.pack(
            fill="both",
            expand=True
        )

        # Intent
        tk.Label(
            content,
            text="INTENT",
            font=("Arial", 12, "bold"),
            anchor="w"
        ).pack(fill="x")

        self.intent_text = tk.Text(
            content,
            height=2,
            wrap="word",
            font=("Arial", 11)
        )

        self.intent_text.pack(
            fill="x",
            pady=(3, 10)
        )

        self.intent_text.config(
            state="disabled"
        )

        # Context
        tk.Label(
            content,
            text="PREVIOUS CONTEXT",
            font=("Arial", 12, "bold"),
            anchor="w"
        ).pack(fill="x")

        self.context_text = tk.Text(
            content,
            height=7,
            wrap="word",
            font=("Arial", 11)
        )

        self.context_text.pack(
            fill="x",
            pady=(3, 10)
        )

        self.context_text.config(
            state="disabled"
        )

        # Customer message
        tk.Label(
            content,
            text="CUSTOMER MESSAGE",
            font=("Arial", 12, "bold"),
            anchor="w"
        ).pack(fill="x")

        self.customer_text = tk.Text(
            content,
            height=6,
            wrap="word",
            font=("Arial", 12)
        )

        self.customer_text.pack(
            fill="x",
            pady=(3, 10)
        )

        self.customer_text.config(
            state="disabled"
        )

        # Historical reply
        tk.Label(
            content,
            text="HISTORICAL APPLESUPPORT REPLY",
            font=("Arial", 12, "bold"),
            anchor="w"
        ).pack(fill="x")

        self.reply_text = tk.Text(
            content,
            height=6,
            wrap="word",
            font=("Arial", 11)
        )

        self.reply_text.pack(
            fill="x",
            pady=(3, 10)
        )

        self.reply_text.config(
            state="disabled"
        )

        # Decision explanation
        tk.Label(
            content,
            text=(
                "ESCALATION DECISION: "
                "YES = send to human support | "
                "NO = AI can handle"
            ),
            font=("Arial", 12, "bold")
        ).pack(
            pady=(5, 10)
        )

        # Buttons
        controls = tk.Frame(
            self.root,
            padx=15,
            pady=10
        )

        controls.pack(fill="x")

        tk.Button(
            controls,
            text="YES - ESCALATE",
            font=("Arial", 12, "bold"),
            width=20,
            height=2,
            command=lambda: self.label_current("yes")
        ).pack(
            side="left",
            padx=10
        )

        tk.Button(
            controls,
            text="NO - HANDLE",
            font=("Arial", 12, "bold"),
            width=20,
            height=2,
            command=lambda: self.label_current("no")
        ).pack(
            side="left",
            padx=10
        )

        tk.Button(
            controls,
            text="SKIP",
            font=("Arial", 12, "bold"),
            width=12,
            height=2,
            command=self.skip_current
        ).pack(
            side="left",
            padx=10
        )

        tk.Button(
            controls,
            text="SAVE & EXIT",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            command=self.save_and_exit
        ).pack(
            side="right",
            padx=10
        )

    # -----------------------------
    # Set text inside disabled box
    # -----------------------------

    def set_text(self, widget, text):

        widget.config(
            state="normal"
        )

        widget.delete(
            "1.0",
            tk.END
        )

        widget.insert(
            tk.END,
            text
        )

        widget.config(
            state="disabled"
        )

    # -----------------------------
    # Show current example
    # -----------------------------

    def show_current(self):

        if self.current_index >= len(self.rows):

            self.progress_label.config(
                text="🎉 All escalation labels are completed!"
            )

            self.set_text(
                self.intent_text,
                ""
            )

            self.set_text(
                self.context_text,
                ""
            )

            self.set_text(
                self.customer_text,
                "Escalation labeling complete."
            )

            self.set_text(
                self.reply_text,
                ""
            )

            return

        row = self.rows[
            self.current_index
        ]

        total = len(self.rows)

        labeled = sum(
            1
            for item in self.rows
            if item.get(
                "escalation",
                ""
            ).strip().lower() in ["yes", "no"]
        )

        self.progress_label.config(
            text=(
                f"Example {self.current_index + 1} / {total}    "
                f"|    Completed: {labeled} / {total}"
            )
        )

        self.set_text(
            self.intent_text,
            row.get(
                "intent",
                ""
            ).strip()
        )

        self.set_text(
            self.context_text,
            row.get(
                "context",
                ""
            ).strip()
            or "(No previous context)"
        )

        self.set_text(
            self.customer_text,
            row.get(
                "customer_message",
                ""
            ).strip()
        )

        self.set_text(
            self.reply_text,
            row.get(
                "historical_agent_reply",
                ""
            ).strip()
            or "(No historical reply)"
        )

    # -----------------------------
    # Label current example
    # -----------------------------

    def label_current(self, decision):

        if self.current_index >= len(self.rows):
            return

        self.rows[
            self.current_index
        ]["escalation"] = decision

        # Save immediately
        self.save_rows()

        self.current_index += 1

        self.find_next_unlabeled()

        self.show_current()

    # -----------------------------
    # Skip
    # -----------------------------

    def skip_current(self):

        if self.current_index >= len(self.rows):
            return

        self.current_index += 1

        self.find_next_unlabeled()

        self.show_current()

    # -----------------------------
    # Save and exit
    # -----------------------------

    def save_and_exit(self):

        self.save_rows()

        messagebox.showinfo(
            "Saved",
            "Escalation labeling progress has been saved."
        )

        self.root.destroy()


# -----------------------------
# Main
# -----------------------------

def main():

    if not INPUT_PATH.exists():

        print(
            f"File not found: {INPUT_PATH}"
        )

        return

    root = tk.Tk()

    app = EscalationLabeler(root)

    root.mainloop()


if __name__ == "__main__":
    main()