# Customer Churn Prediction using XGBoost

A machine learning project for predicting whether a telecommunications customer is likely to churn. The project covers the complete machine learning workflow, from exploratory data analysis and preprocessing to model comparison, XGBoost optimization, model interpretation, and deployment through a FastAPI REST API.

---

## Project Overview

Customer churn prediction is a binary classification problem where the objective is to identify customers who are likely to discontinue their service.

This project uses the **Telco Customer Churn** dataset and evaluates multiple machine learning approaches before selecting a final XGBoost model.

The project focuses not only on achieving a high accuracy score, but also on understanding:

* Which customer characteristics are associated with churn
* How different machine learning algorithms perform
* Why XGBoost performs differently from simpler models
* How hyperparameter tuning affects model performance
* How classification thresholds affect precision and recall
* How the trained model can be exposed through an API

---

## Project Objectives

The main objectives of this project are:

1. Perform exploratory data analysis on customer churn data.
2. Clean and preprocess the dataset for machine learning.
3. Establish baseline models using different classification algorithms.
4. Develop an XGBoost classification model.
5. Apply early stopping and hyperparameter tuning.
6. Compare model performance using multiple evaluation metrics.
7. Analyze important features contributing to predictions.
8. Evaluate different classification thresholds.
9. Save the final trained model as a reusable pipeline.
10. Deploy the model through a FastAPI REST API.

---

## Dataset

The dataset contains **7,043 customer records** and **21 original columns**.

The target variable is:

```text
Churn
```

where:

```text
Yes → 1
No  → 0
```

### Main Features

| Feature            | Description                              |
| ------------------ | ---------------------------------------- |
| `gender`           | Customer gender                          |
| `SeniorCitizen`    | Whether the customer is a senior citizen |
| `Partner`          | Whether the customer has a partner       |
| `Dependents`       | Whether the customer has dependents      |
| `tenure`           | Number of months the customer has stayed |
| `PhoneService`     | Whether phone service is subscribed      |
| `MultipleLines`    | Multiple phone lines subscription        |
| `InternetService`  | Type of internet service                 |
| `OnlineSecurity`   | Online security subscription             |
| `OnlineBackup`     | Online backup subscription               |
| `DeviceProtection` | Device protection subscription           |
| `TechSupport`      | Technical support subscription           |
| `StreamingTV`      | Streaming TV subscription                |
| `StreamingMovies`  | Streaming movies subscription            |
| `Contract`         | Contract type                            |
| `PaperlessBilling` | Paperless billing status                 |
| `PaymentMethod`    | Payment method                           |
| `MonthlyCharges`   | Monthly customer charges                 |
| `TotalCharges`     | Total customer charges                   |
| `Churn`            | Target variable                          |

`customerID` was removed because it is an identifier rather than a meaningful predictive feature.

---

## Project Workflow

```text
Dataset
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Cleaning & Preprocessing
   │
   ▼
Baseline Models
   │
   ├── Logistic Regression
   ├── Decision Tree
   └── Random Forest
   │
   ▼
XGBoost
   │
   ▼
Early Stopping
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Model Analysis
   │
   ├── Model Comparison
   ├── Feature Importance
   └── Threshold Analysis
   │
   ▼
Final XGBoost Pipeline
   │
   ▼
FastAPI
```

---

# Exploratory Data Analysis

The dataset initially contained:

* 7,043 rows
* 21 columns
* 18 categorical features
* 2 integer features
* 1 floating-point feature

`TotalCharges` was initially interpreted as a string despite containing numerical values. After converting it to a numeric type, 11 missing values were identified.

These records corresponded to customers with `tenure = 0`, indicating newly registered customers with no accumulated charges. These missing values were therefore replaced with `0`.

No duplicate customer records were identified.

---

## Important EDA Findings

Several patterns were identified during exploratory analysis.

### Contract Type

Month-to-month customers showed substantially higher churn than customers on one-year or two-year contracts.

### Internet Service

Fiber-optic customers showed considerably higher churn compared with DSL and customers without internet service.

### Tenure

Customers with longer tenure generally showed lower churn rates.

### Monthly Charges

Higher monthly charges were associated with higher churn.

### Technical and Security Services

Customers without services such as:

* Online Security
* Tech Support
* Device Protection
* Online Backup

showed higher churn proportions.

### Payment Method

Electronic-check customers showed noticeably higher churn compared with customers using automatic payment methods.

These relationships represent **associations observed in the dataset** and should not be interpreted as causal relationships.

---

# Preprocessing

The preprocessing workflow included:

1. Removing `customerID`.
2. Converting `TotalCharges` to numeric.
3. Replacing the 11 `TotalCharges` values corresponding to `tenure = 0` with `0`.
4. Encoding the target:

   * `Yes → 1`
   * `No → 0`
5. Separating numerical and categorical features.
6. Applying one-hot encoding to categorical variables.
7. Passing numerical variables through without categorical encoding.

The preprocessing transformation was incorporated into the final machine learning pipeline so that the same transformation can be applied during inference.

---

# Baseline Models

Three baseline classification models were evaluated before developing the final XGBoost model:

* Logistic Regression
* Decision Tree
* Random Forest

The purpose of the baseline stage was to establish reference performance and understand whether the more advanced XGBoost model provided a meaningful improvement.

---

## Baseline Results

| Model               |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Decision Tree       |     72.25% |     47.81% |     49.73% |     48.75% |     65.04% |
| Random Forest       |     78.85% |     63.10% |     48.93% |     55.12% |     81.80% |
| Logistic Regression |     80.41% |     65.31% | **55.88%** | **60.23%** |     84.26% |
| **XGBoost**         | **80.48%** | **66.78%** |     52.67% |     58.89% | **84.84%** |

The Decision Tree performed the weakest.

Random Forest improved substantially over a single Decision Tree, demonstrating the benefit of ensemble learning.

Logistic Regression performed surprisingly strongly and achieved the highest recall and F1 score among the models at the default threshold.

XGBoost achieved the highest ROC-AUC, accuracy, and precision.

---

# XGBoost Model

XGBoost was selected for further experimentation because of its strong baseline performance and ability to model nonlinear relationships and feature interactions.

The final tuned model used the following configuration:

```python
XGBClassifier(
    n_estimators=198,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=10,
    gamma=0.1,
    reg_lambda=1,
    random_state=42,
    eval_metric="logloss"
)
```

Early stopping was used during the model-development stage to determine an appropriate number of boosting iterations and reduce unnecessary training.

---

# Final XGBoost Evaluation

The final XGBoost model was evaluated on the test set using the default classification threshold of `0.50`.

### Results

```text
Accuracy  : 80.48%
Precision : 66.78%
Recall    : 52.67%
F1 Score  : 58.89%
ROC-AUC   : 84.84%
```

### Confusion Matrix

```text
[[937   98]
 [177  197]]
```

This corresponds to:

|            | Predicted No | Predicted Yes |
| ---------- | -----------: | ------------: |
| Actual No  |          937 |            98 |
| Actual Yes |          177 |           197 |

The model correctly identified 197 churners while missing 177 actual churners.

---

# Decision Threshold Analysis

The default classification threshold was compared with alternative thresholds using the validation set.

| Threshold | Precision | Recall |        F1 |
| --------: | --------: | -----: | --------: |
|      0.20 |     0.484 |  0.856 |     0.618 |
|      0.25 |     0.515 |  0.803 |     0.627 |
|      0.30 |     0.554 |  0.766 |     0.643 |
|      0.35 |     0.605 |  0.722 | **0.659** |
|      0.40 |     0.617 |  0.652 |     0.634 |
|      0.45 |     0.667 |  0.582 |     0.621 |
|  **0.50** | **0.697** |  0.515 |     0.592 |
|      0.55 |     0.726 |  0.425 |     0.536 |
|      0.60 |     0.727 |  0.365 |     0.486 |

Although a threshold of `0.35` produced higher recall and F1 on the validation data, it also generated substantially more false positives when evaluated on the test set.

For this project, the default threshold of **0.50** was retained because it provides a more conservative operating point with higher precision and accuracy.

Therefore:

```text
Final classification threshold = 0.50
```

---

# Feature Importance

Feature importance from the final XGBoost model showed that several features contributed strongly to the model's predictive decisions.

### Top Features

| Rank | Feature                          | Importance |
| ---: | -------------------------------- | ---------: |
|    1 | `Contract_Month-to-month`        |     0.4945 |
|    2 | `InternetService_Fiber optic`    |     0.1349 |
|    3 | `OnlineSecurity_No`              |     0.0544 |
|    4 | `TechSupport_No`                 |     0.0394 |
|    5 | `InternetService_DSL`            |     0.0209 |
|    6 | `Contract_Two year`              |     0.0201 |
|    7 | `tenure`                         |     0.0201 |
|    8 | `StreamingMovies_Yes`            |     0.0197 |
|    9 | `Contract_One year`              |     0.0172 |
|   10 | `PaymentMethod_Electronic check` |     0.0164 |

The model relied particularly strongly on contract type, especially the month-to-month category.

Feature importance indicates **predictive contribution within the trained model**, not causation.

---

# Final Model Selection

The final model was selected based on its overall performance and suitability for the prediction task.

XGBoost achieved:

```text
ROC-AUC  = 84.84%
Accuracy = 80.48%
Precision = 66.78%
```

It achieved the highest ROC-AUC among the evaluated models, indicating strong discrimination between churn and non-churn customers.

However, Logistic Regression achieved slightly better recall and F1 at the default threshold. This demonstrates that model selection depends on the objective and evaluation metric rather than simply choosing the most complex algorithm.

For this project, **XGBoost was selected as the final model**.

---

# Deployment

The trained model is packaged as a reusable preprocessing and prediction pipeline.

The deployment architecture is:

```text
Customer Input
      │
      ▼
   FastAPI
      │
      ▼
Pydantic Validation
      │
      ▼
Preprocessing Pipeline
      │
      ▼
    XGBoost
      │
      ▼
Churn Probability
      │
      ▼
Threshold = 0.50
      │
      ▼
Churn / No Churn
```

The API exposes a prediction endpoint:

```text
POST /predict
```

Example response:

```json
{
  "churn_probability": 0.73,
  "prediction": "Yes"
}
```

The frontend is intentionally not included in the current version of the project.

---

# Project Structure

```text
customer-churn-xgboost/
│
├── data/
│   └── cleaned_churn.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_xgboost.ipynb
│   └── 04_model_analysis.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── churn_pipeline.pkl
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── model_service.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd customer-churn-xgboost
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Train the Model

To train the final model and generate the saved pipeline:

```bash
python src/train.py
```

The trained pipeline will be saved to:

```text
models/churn_pipeline.pkl
```

---

# Test a Prediction

After training:

```bash
python src/predict.py
```

This loads the saved pipeline and performs a prediction on a sample customer.

---

# Run the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available through FastAPI's automatically generated Swagger interface at:

```text
http://127.0.0.1:8000/docs
```

---

# Limitations

Several limitations should be considered:

* The model is trained on a single telecommunications churn dataset.
* The dataset represents historical customer behavior and may not generalize to other telecommunications companies.
* Feature importance does not establish causal relationships.
* The model does not incorporate temporal customer behavior beyond the available features.
* The default 0.50 threshold was selected as a conservative operating point without an explicit business cost matrix.
* The model's recall indicates that a significant portion of actual churners remain undetected.

---

# Future Improvements

Potential future improvements include:

* Cost-sensitive learning based on the financial cost of false negatives and false positives.
* More systematic hyperparameter optimization.
* Probability calibration.
* SHAP-based model explainability.
* Cross-validation for more robust performance estimation.
* Monitoring model performance after deployment.
* Adding a frontend interface for non-technical users.
* Containerizing the API with Docker.
* Deploying the API to a cloud platform.
* Adding automated tests for the prediction pipeline and API.

---

# Conclusion

This project demonstrates an end-to-end customer churn prediction workflow using machine learning.

Several classification models were evaluated, including Logistic Regression, Decision Tree, Random Forest, and XGBoost. XGBoost achieved the strongest overall discrimination with a test ROC-AUC of **0.8484** and was therefore selected as the final model.

The analysis also demonstrated that model evaluation cannot rely on accuracy alone. Precision, recall, F1 score, ROC-AUC, confusion matrices, and decision thresholds provide different perspectives on model performance.

The final system packages preprocessing and XGBoost into a reusable prediction pipeline and exposes the model through a FastAPI REST API, providing the foundation for a production-oriented machine learning application.
