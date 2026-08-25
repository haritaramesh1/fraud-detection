import pandas as pd

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/PS_20174392719_1491204439457_log.csv"
)


# ============================================================
# 2. INVESTIGATION STATE
# ============================================================

class InvestigationState(TypedDict):

    # Transaction being investigated.
    transaction: dict

    # Recent transactions belonging to the account.
    history: list

    # SHAP reasons explaining the model decision.
    shap_reasons: list

    # Draft investigation note.
    note: str

    # Human decision.
    decision: str


# ============================================================
# 3. GATHER HISTORY NODE
# ============================================================

def gather_history(
    state: InvestigationState,
) -> InvestigationState:

    print(
        "\n[1/3] Gathering account history..."
    )

    # Load the dataset.

    df = pd.read_csv(
        DATA_PATH
    )


    transaction = state[
        "transaction"
    ]


    # Get the originating account.

    account = transaction[
        "nameOrig"
    ]


    # Find recent transactions from this account.

    history = df[
        df["nameOrig"] == account
    ].sort_values(
        "step",
        ascending=False,
    ).head(10)


    # Convert to ordinary Python dictionaries
    # so LangGraph can carry the information.

    history_records = (
        history[
            [
                "step",
                "type",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "isFraud",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )


    print(
        "Account:",
        account
    )

    print(
        "Recent transactions found:",
        len(history_records)
    )


    return {
        **state,
        "history": history_records,
    }


# ============================================================
# 4. DRAFT INVESTIGATION NOTE
# ============================================================

def draft_note(
    state: InvestigationState,
) -> InvestigationState:

    print(
        "\n[2/3] Drafting investigation note..."
    )


    transaction = state[
        "transaction"
    ]


    history = state[
        "history"
    ]


    shap_reasons = state[
        "shap_reasons"
    ]


    # Count recent transactions.

    transaction_count = len(
        history
    )


    # Calculate recent transaction value.

    recent_amount = sum(
        float(row["amount"])
        for row in history
    )


    # Build human-readable SHAP explanation.

    if shap_reasons:

        reasons_text = "; ".join(
            shap_reasons[:3]
        )

    else:

        reasons_text = (
            "No SHAP explanation supplied."
        )


    # Create a short investigation note.
    #
    # This is deliberately a deterministic draft for now.
    # Later we can replace this section with a Gemini call.

    note = f"""
FIN SENTINEL INVESTIGATION NOTE

Transaction:
- Type: {transaction["type"]}
- Amount: ₹{float(transaction["amount"]):,.2f}
- Account: {transaction["nameOrig"]}
- Step: {transaction["step"]}

Recent account activity:
- Transactions found: {transaction_count}
- Recent transaction value: ₹{recent_amount:,.2f}

Model explanation:
- {reasons_text}

Recommendation:
Review the transaction and account history before taking action.
No automatic blocking has been performed.
""".strip()


    return {
        **state,
        "note": note,
    }


# ============================================================
# 5. HUMAN REVIEW NODE
# ============================================================

def human_review(
    state: InvestigationState,
) -> InvestigationState:

    print(
        "\n[3/3] HUMAN REVIEW"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        state["note"]
    )

    print(
        "=" * 60
    )


    # IMPORTANT:
    #
    # The system stops here and asks a human.
    #
    # It does NOT automatically block the transaction.

    decision = input(
        "\nApprove investigation? "
        "(y = approve / n = reject): "
    ).strip().lower()


    if decision == "y":

        final_decision = "approved"

    elif decision == "n":

        final_decision = "rejected"

    else:

        final_decision = "invalid"


    print(
        "\nHuman decision:",
        final_decision
    )


    return {
        **state,
        "decision": final_decision,
    }


# ============================================================
# 6. BUILD LANGGRAPH
# ============================================================

print(
    "\nBuilding investigation graph..."
)


builder = StateGraph(
    InvestigationState
)


# Add the three nodes.

builder.add_node(
    "gather_history",
    gather_history,
)

builder.add_node(
    "draft_note",
    draft_note,
)

builder.add_node(
    "human_review",
    human_review,
)


# Define the workflow.

builder.add_edge(
    START,
    "gather_history",
)

builder.add_edge(
    "gather_history",
    "draft_note",
)

builder.add_edge(
    "draft_note",
    "human_review",
)

builder.add_edge(
    "human_review",
    END,
)


# Compile the graph.

graph = builder.compile()


# ============================================================
# 7. DEMO TRANSACTION
# ============================================================

# This transaction comes from the PaySim dataset.
#
# We use a real transaction so the investigation workflow
# can inspect its actual account history.

demo_transaction = {
    "step": 400,
    "type": "TRANSFER",
    "amount": 274184.08,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 55219.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 100000.0,
    "newbalanceDest": 374184.08,
    "isFlaggedFraud": 0,
}


# ============================================================
# 8. SHAP REASONS
# ============================================================

# These represent the kinds of explanations produced
# by Stage 4.
#
# Later, we will connect this automatically to the
# actual SHAP output.

demo_shap_reasons = [
    "origin account balance strongly increased fraud risk",
    "destination balance strongly increased fraud risk",
    "transaction amount strongly increased fraud risk",
]


# ============================================================
# 9. INITIAL STATE
# ============================================================

initial_state: InvestigationState = {

    "transaction": demo_transaction,

    "history": [],

    "shap_reasons": demo_shap_reasons,

    "note": "",

    "decision": "",
}


# ============================================================
# 10. RUN INVESTIGATION
# ============================================================

print(
    "\n" + "=" * 60
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


final_state = graph.invoke(
    initial_state
)


# ============================================================
# 11. FINAL RESULT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "INVESTIGATION COMPLETE"
)

print(
    "=" * 60
)

print(
    "Human decision:",
    final_state["decision"]
)

print(
    "\nThe transaction was NOT automatically blocked."
)

print(
    "A human made the final decision."
)