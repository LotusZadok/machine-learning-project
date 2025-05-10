import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df = pd.read_csv('./data/processed/telco_customer_churn_encoded.csv')

features = [
    'PaymentMethod_Credit card (automatic)', 'SeniorCitizen', 'Contract_One year',
    'StreamingMovies_Yes', 'Dependents_Yes', 'OnlineSecurity_Yes',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'gender_Male',
    'MonthlyCharges', 'OnlineBackup_Yes', 'InternetService_Fiber optic',
    'MultipleLines_Yes', 'TechSupport_Yes', 'DeviceProtection_Yes',
    'Contract_Two year', 'tenure', 'StreamingTV_Yes', 'PaperlessBilling_Yes'
]

X = df[features]
y = df['Churn_Yes']

# splitting the data and applying smote
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Balance after SMOTE:", np.bincount(y_train_res))

# search space for hyperparameters
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

param_dist = {
    'n_estimators':     np.arange(200, 1201, 200),         # 200‑1200
    'learning_rate':    np.linspace(0.01, 0.3, 10),        # 0.01‑0.30
    'max_depth':        np.arange(3, 11),                  # 3‑10
    'min_child_weight': np.arange(1, 7),                   # 1‑6
    'subsample':        np.linspace(0.6, 1.0, 5),          # 0.6‑1.0
    'colsample_bytree': np.linspace(0.6, 1.0, 5),          # 0.6‑1.0
    'gamma':            np.linspace(0, 0.4, 5),            # 0‑0.4
    'reg_alpha':        [0, 0.01, 0.1, 1],                 # L1
    'reg_lambda':       [0.5, 1, 2, 5, 10]                 # L2
}

search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=200,
    scoring='roc_auc',
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train_res, y_train_res)

best_model = search.best_estimator_
print("\nBest params:", search.best_params_)
print("Best CV AUC:", search.best_score_)

# final evaluation

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
print("\nTest AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'])
plt.title("XGBoost (Iteration 2) – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# ROC & PR curves
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.show()
PrecisionRecallDisplay.from_predictions(y_test, y_prob)
plt.show()

joblib.dump(best_model, './models/xgboost_iteration_2_refined.pkl')
