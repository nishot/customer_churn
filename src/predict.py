
# src/predict.py

import joblib
import pandas as pd




MODEL_PATH = "models/churn_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")



customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.50,
    "TotalCharges": 966.00
}


# Convert dictionary into a DataFrame
customer_df = pd.DataFrame([customer])



churn_probability = pipeline.predict_proba(customer_df)[0][1]




threshold = 0.50

prediction = int(churn_probability >= threshold)




print("\nCustomer Prediction")
print("=" * 40)

print(f"Churn Probability : {churn_probability:.2%}")

if prediction == 1:
    print("Prediction         : CHURN")
else:
    print("Prediction         : NO CHURN")
