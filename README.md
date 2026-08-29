STEP 1
Choose + understand dataset
        ↓
STEP 2
EDA
        ↓
STEP 3
Preprocessing
        ↓
STEP 4
Baseline models
        ↓
STEP 5
XGBoost
        ↓
STEP 6
Early stopping
        ↓
STEP 7
Tuning
        ↓
STEP 8
Model analysis
        ↓
STEP 9
Save model
        ↓
STEP 10
FastAPI
        ↓
STEP 11
Frontend
        ↓
STEP 12
Deployment




customer-churn-xgboost/
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_xgboost.ipynb
│   └── 04_model_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── churn_pipeline.pkl
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── model_service.py
│
├── frontend/
│
├── requirements.txt
├── .gitignore
└── README.md