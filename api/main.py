

from fastapi import FastAPI

from .schemas import CustomerData
from .services import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="XGBoost-based customer churn prediction service",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    result = predict_churn(
        customer.model_dump()
    )

    return result

