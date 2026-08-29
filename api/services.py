
# api/model_service.py

import joblib
import pandas as pd


MODEL_PATH = "models/churn_pipeline.pkl"

# Load model once when the API starts
pipeline = joblib.load(MODEL_PATH)

THRESHOLD = 0.50


def predict_churn(customer_data: dict):

    # Convert dictionary to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Get probability of churn
    churn_probability = pipeline.predict_proba(customer_df)[0][1]

    # Apply decision threshold
    prediction = int(churn_probability >= THRESHOLD)

    return {
        "churn_probability": float(churn_probability),
        "prediction": "Yes" if prediction == 1 else "No"
    }

