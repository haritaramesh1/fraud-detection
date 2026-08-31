"""
FinSentinel Investigator
------------------------

Human-in-the-loop fraud investigation workflow.

Flow:

1. Gather recent account history
2. Draft an investigation note using Gemini
3. If Gemini is unavailable, create a safe fallback note
4. Ask a human reviewer to approve/reject
5. Never automatically block a transaction

Gemini is used ONLY to draft/summarize the investigation note.
The human makes the final decision.
"""

import os
import time
from typing import TypedDict, Optional

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

# Load variables from .env
load_dotenv()

# Gemini API key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini model
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

# Dataset location
DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"

# Account we are investigating
ACCOUNT_ID = "C1231006815"

# Transaction being investigated
TRANSACTION = {
    "step": 400,
    "type": "TRANSFER",
    "amount": 274184.08,
    "oldbalanceOrg": 55219.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 100000.0,
    "newbalanceDest": 374184.08,
    "isFlaggedFraud": 0,
}

# Number of recent transactions to provide to Gemini
HISTORY_LIMIT = 10

# Gemini retry settings
MAX_GEMINI_RETRIES = 3
RETRY_DELAY_SECONDS = 3


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("Gemini client initialized.")

    except Exception as e:
        print(f"Warning: Gemini client could not be initialized: {e}")

else:
    print("Warning: GEMINI_API_KEY was not found.")
    print("Gemini will be skipped and fallback investigation notes will be used.")


# ============================================================
# LANGGRAPH STATE
# ============================================================

class InvestigationState(TypedDict, total=False):
    """
    State shared between all LangGraph nodes.
    """

    account_id: str

    transaction: dict

    history: list

    shap_reasons: list

    investigation_note: str

    gemini_used: bool

    human_decision: str


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    """
    Load the PaySim transaction dataset.
    """

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {len(df):,} rows")

    return df


# ============================================================
# SHAP / MODEL REASONS
# ============================================================

def get_model_reasons():
    """
    Return the important model factors discovered during SHAP analysis.

    These are based on the actual SHAP analysis performed earlier
    in the FinSentinel project.
    """

    return [
        "origin account balance strongly increased fraud risk",
        "destination balance strongly increased fraud risk",
        "transaction amount strongly increased fraud risk",
    ]


# ============================================================
# NODE 1 — GATHER ACCOUNT HISTORY
# ============================================================

def gather_history(state: InvestigationState):
    """
    Gather recent transactions belonging to the investigated account.
    """

    print("\n[1/3] Gathering account history...")

    account_id = state["account_id"]

    print(f"Account: {account_id}")

    try:
        df = load_dataset()

        # PaySim contains origin and destination account IDs.
        #
        # We check both fields because an account can appear
        # as either the sender or receiver.
        history = df[
            (df["nameOrig"] == account_id)
            | (df["nameDest"] == account_id)
        ].copy()

        # Sort by time.
        history = history.sort_values(
            "step",
            ascending=False
        )

        # Keep only the most recent transactions.
        history = history.head(HISTORY_LIMIT)

        records = []

        for _, row in history.iterrows():

            records.append(
                {
                    "step": int(row["step"]),
                    "type": str(row["type"]),
                    "amount": float(row["amount"]),
                    "nameOrig": str(row["nameOrig"]),
                    "nameDest": str(row["nameDest"]),
                    "isFraud": int(row["isFraud"])
                    if "isFraud" in row
                    else None,
                }
            )

        print(
            f"Recent transactions found: {len(records)}"
        )

        state["history"] = records

        return state

    except Exception as e:

        print(
            f"Could not load account history: {e}"
        )

        state["history"] = []

        return state


# ============================================================
# FALLBACK INVESTIGATION NOTE
# ============================================================

def build_fallback_note(
    state: InvestigationState
) -> str:
    """
    Create an investigation note without an LLM.

    This makes the system resilient when Gemini is unavailable.
    """

    transaction = state["transaction"]

    history = state.get(
        "history",
        []
    )

    reasons = state.get(
        "shap_reasons",
        []
    )

    amount = transaction["amount"]

    note = []

    note.append(
        "FinSentinel Investigation Note"
    )

    note.append("")

    note.append("Transaction:")
    note.append(
        f"- Type: {transaction['type']}"
    )
    note.append(
        f"- Amount: ₹{amount:,.2f}"
    )
    note.append(
        f"- Account: {state['account_id']}"
    )
    note.append(
        f"- Step: {transaction['step']}"
    )

    note.append("")

    note.append(
        "Recent account activity:"
    )

    note.append(
        f"- Transactions found: {len(history)}"
    )

    if history:

        recent = history[0]

        note.append(
            f"- Most recent transaction: "
            f"₹{recent['amount']:,.2f}"
        )

        note.append(
            f"- Most recent transaction type: "
            f"{recent['type']}"
        )

    else:

        note.append(
            "- No recent account history was found."
        )

    note.append("")

    note.append(
        "Model explanation:"
    )

    for reason in reasons:

        note.append(
            f"- {reason}"
        )

    note.append("")

    note.append(
        "Recommendation:"
    )

    note.append(
        "Review the transaction and account "
        "history before taking action."
    )

    note.append(
        "No automatic blocking has been performed."
    )

    return "\n".join(note)


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    state: InvestigationState
) -> str:
    """
    Build the prompt sent to Gemini.

    Gemini is given structured transaction information,
    account history, and model explanations.
    """

    transaction = state["transaction"]

    history = state.get(
        "history",
        []
    )

    reasons = state.get(
        "shap_reasons",
        []
    )

    history_text = "\n".join(
        [
            (
                f"- Step {item['step']}: "
                f"{item['type']} "
                f"₹{item['amount']:,.2f}"
            )
            for item in history
        ]
    )

    reasons_text = "\n".join(
        [
            f"- {reason}"
            for reason in reasons
        ]
    )

    prompt = f"""
You are assisting a financial fraud investigator.

You are NOT allowed to make the final fraud decision.

Your task is ONLY to draft a concise investigation note
for a human reviewer.

Transaction:
- Account: {state['account_id']}
- Type: {transaction['type']}
- Amount: ₹{transaction['amount']:,.2f}
- Step: {transaction['step']}
- Origin balance before: ₹{transaction['oldbalanceOrg']:,.2f}
- Origin balance after: ₹{transaction['newbalanceOrig']:,.2f}
- Destination balance before: ₹{transaction['oldbalanceDest']:,.2f}
- Destination balance after: ₹{transaction['newbalanceDest']:,.2f}
- Flagged by original dataset rule: {transaction['isFlaggedFraud']}

Recent account activity:
{history_text if history_text else "- No recent activity available."}

Model explanation:
{reasons_text}

Write an investigation note with these sections:

1. Transaction
2. Recent account activity
3. Model explanation
4. Recommendation

Rules:
- Do not claim the transaction is definitely fraud.
- Do not invent facts.
- Do not invent additional transactions.
- Do not recommend automatic blocking.
- State that a human reviewer must make the final decision.
- Keep the note concise and professional.
"""

    return prompt


# ============================================================
# NODE 2 — DRAFT INVESTIGATION NOTE
# ============================================================

def draft_note(state: InvestigationState):
    """
    Ask Gemini to draft the investigation note.

    If Gemini fails temporarily, retry several times.

    If Gemini still cannot respond, use the deterministic
    fallback note so the investigation workflow continues.
    """

    print(
        "\n[2/3] Drafting investigation note with Gemini..."
    )

    # Default to false.
    state["gemini_used"] = False

    # If Gemini is unavailable, use fallback.
    if gemini_client is None:

        print(
            "Gemini is not configured."
        )

        print(
            "Using fallback investigation note..."
        )

        state["investigation_note"] = (
            build_fallback_note(state)
        )

        return state

    prompt = build_gemini_prompt(state)

    # Try Gemini multiple times.
    for attempt in range(
        1,
        MAX_GEMINI_RETRIES + 1
    ):

        try:

            print(
                f"Gemini attempt "
                f"{attempt}/{MAX_GEMINI_RETRIES}..."
            )

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            # Gemini response text.
            generated_note = response.text

            # Protect against empty responses.
            if not generated_note:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            state["investigation_note"] = (
                generated_note.strip()
            )

            state["gemini_used"] = True

            print(
                "Gemini investigation note generated successfully."
            )

            return state

        except Exception as e:

            print(
                f"Gemini attempt {attempt} failed:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            # Retry if attempts remain.
            if attempt < MAX_GEMINI_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    # All Gemini attempts failed.
    print("")
    print(
        "Gemini is temporarily unavailable."
    )

    print(
        "Using deterministic fallback note."
    )

    state["investigation_note"] = (
        build_fallback_note(state)
    )

    state["gemini_used"] = False

    return state


# ============================================================
# NODE 3 — HUMAN REVIEW
# ============================================================

def human_review(state: InvestigationState):
    """
    Human-in-the-loop decision.

    The system does NOT automatically block the transaction.
    """

    print(
        "\n[3/3] HUMAN REVIEW"
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FIN SENTINEL INVESTIGATION NOTE"
    )

    print(
        "=" * 60
    )

    print(
        state["investigation_note"]
    )

    print(
        "\n"
        + "=" * 60
    )

    if state.get("gemini_used"):

        print(
            "Note drafted with Gemini."
        )

    else:

        print(
            "Note generated using fallback logic."
        )

    print(
        "The AI does NOT make the final decision."
    )

    print(
        "=" * 60
    )

    # Ask the human reviewer.
    while True:

        decision = input(
            "\nApprove investigation? "
            "(y = approve / n = reject): "
        ).strip().lower()

        if decision in ("y", "yes"):

            state["human_decision"] = "approved"

            break

        if decision in ("n", "no"):

            state["human_decision"] = "rejected"

            break

        print(
            "Please enter y or n."
        )

    return state


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():
    """
    Build the LangGraph workflow.

    LangGraph connects the three investigation stages
    and passes the shared state between them.
    """

    graph_builder = StateGraph(
        InvestigationState
    )

    # Add investigation nodes.
    graph_builder.add_node(
        "gather_history",
        gather_history
    )

    graph_builder.add_node(
        "draft_note",
        draft_note
    )

    graph_builder.add_node(
        "human_review",
        human_review
    )

    # Define workflow order.
    graph_builder.add_edge(
        START,
        "gather_history"
    )

    graph_builder.add_edge(
        "gather_history",
        "draft_note"
    )

    graph_builder.add_edge(
        "draft_note",
        "human_review"
    )

    graph_builder.add_edge(
        "human_review",
        END
    )

    return graph_builder.compile()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("")
    print(
        "Building investigation graph..."
    )

    graph = build_graph()

    print("")
    print(
        "=" * 60
    )

    print(
        "FINSENTINEL INVESTIGATOR"
    )

    print(
        "=" * 60
    )

    print(
        "Starting investigation..."
    )

    # Model/SHAP explanations discovered earlier.
    shap_reasons = get_model_reasons()

    # Initial LangGraph state.
    initial_state: InvestigationState = {

        "account_id": ACCOUNT_ID,

        "transaction": TRANSACTION,

        "history": [],

        "shap_reasons": shap_reasons,

        "investigation_note": "",

        "gemini_used": False,

        "human_decision": "",
    }

    # Execute the graph.
    final_state = graph.invoke(
        initial_state
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("")
    print(
        "=" * 60
    )

    print(
        "INVESTIGATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Human decision: "
        f"{final_state['human_decision']}"
    )

    print("")

    if final_state["human_decision"] == "approved":

        print(
            "Investigation approved by human reviewer."
        )

    else:

        print(
            "Investigation rejected by human reviewer."
        )

    print(
        "The transaction was NOT automatically blocked."
    )

    print(
        "A human made the final decision."
    )

    print(
        "=" * 60
    )