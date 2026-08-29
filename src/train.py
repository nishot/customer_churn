

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier




DATA_PATH = "data/cleaned/churnData(cleaned).csv"
MODEL_PATH = "models/churn_pipeline.pkl"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)



X = df.drop(columns=["Churn"])
y = df["Churn"]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])



categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)



xgb_model = XGBClassifier(
    n_estimators=198,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=10,
    gamma=0.1,
    reg_lambda=1,
    random_state=42,
    eval_metric="logloss"
)



churn_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)



print("\nTraining final XGBoost pipeline...")

churn_pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")




os.makedirs("models", exist_ok=True)

joblib.dump(
    churn_pipeline,
    MODEL_PATH
)

print(f"\nPipeline saved to: {MODEL_PATH}")
