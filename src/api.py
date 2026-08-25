import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# 1. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FinSentinel Fraud Detection API",
    description="Real-time fraud scoring API.",
    version="1.0.0",
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/finsentinel_model.joblib"

PREPROCESSOR_PATH = (
    "models/finsentinel_preprocessor.joblib"
)


print("Loading FinSentinel model...")

model = joblib.load(
    MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

print("Model loaded successfully.")


# ============================================================
# 3. TRANSACTION INPUT SCHEMA
# ============================================================

class Transaction(BaseModel):

    step: int

    type: str

    amount: float

    oldbalanceOrg: float

    newbalanceOrig: float

    oldbalanceDest: float

    newbalanceDest: float

    isFlaggedFraud: int = 0


# ============================================================
# 4. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "FinSentinel",
        "status": "online",
        "endpoint": "/score",
    }


# ============================================================
# 5. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ============================================================
# 6. SCORE ENDPOINT
# ============================================================

@app.post("/score")
def score_transaction(
    transaction: Transaction,
):

    # Convert the incoming JSON object into a
    # pandas DataFrame.

    transaction_df = pd.DataFrame(
        [
            transaction.model_dump()
        ]
    )


    # Apply the exact preprocessing used during training.

    encoded = preprocessor.transform(
        transaction_df
    )


    # Generate fraud probability.

    probability = model.predict_proba(
        encoded
    )[0, 1]


    # Money-optimized threshold from Stage 3.

    threshold = 0.0060


    # Decide whether to flag the transaction.

    flagged = (
        probability >= threshold
    )


    return {
        "fraud_probability": round(
            float(probability),
            6,
        ),
        "threshold": threshold,
        "flagged": bool(flagged),
    }