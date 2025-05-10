# ----------------------------------------------------------
# File: src/iteration_3/xgboost_optuna.py
# Purpose: Iteration 3 – XGBoost with Optuna hyper‑parameter tuning
# ----------------------------------------------------------
import optuna
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df = pd.read_csv("./data/processed/telco_customer_churn_encoded.csv")

REFINED_FEATURES = [
    'PaymentMethod_Credit card (automatic)', 'SeniorCitizen',
    'Contract_One year', 'StreamingMovies_Yes', 'Dependents_Yes',
    'OnlineSecurity_Yes', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check', 'gender_Male', 'MonthlyCharges',
    'OnlineBackup_Yes', 'InternetService_Fiber optic',
    'MultipleLines_Yes', 'TechSupport_Yes', 'DeviceProtection_Yes',
    'Contract_Two year', 'tenure', 'StreamingTV_Yes', 'PaperlessBilling_Yes'
]

X = df[REFINED_FEATURES]
y = df["Churn_Yes"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# balance classes with SMOTE + standardise
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_scl  = scaler.transform(X_test)

print("Balanced class distribution:", np.bincount(y_train_res))

# optuna objective
def objective(trial):
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "n_jobs":           -1,
        "random_state":     42,
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "n_estimators":     trial.suggest_int("n_estimators", 300, 1200, step=100),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
    }

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.2,
        stratify=y_train_res, random_state=42
    )

    model = XGBClassifier(**params)
    model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

    preds = model.predict_proba(X_val)[:, 1]
    auc   = roc_auc_score(y_val, preds)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\nBest validation AUC :", study.best_value)
print("Best hyper‑parameters:", study.best_params)

# train best model on full resampled training set
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "random_state": 42
})

best_model = XGBClassifier(**best_params)
best_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_train_res, y_train_res)],
    verbose=True
)

# evaluate on hold‑out test set
y_pred = best_model.predict(X_test_scl)
y_prob = best_model.predict_proba(X_test_scl)[:, 1]

print("\nTest AUC :", roc_auc_score(y_test, y_prob))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, cmap="viridis", fmt="d",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)
plt.title("XGBoost (Iteration 3 – Optuna) · Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()


joblib.dump(best_model, "./models/xgboost_iteration_3_optuna.pkl")
print("\nModel saved to ../../models/xgboost_iteration_3_optuna.pkl")
