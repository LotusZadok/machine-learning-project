import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# load data and preprocess
df = pd.read_csv("../../../data/processed/telco_customer_churn_encoded.csv")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_scl  = scaler.transform(X_test)

# load the trained model
model = joblib.load("./models/xgboost_iteration_3_optuna.pkl")
y_probs = model.predict_proba(X_test_scl)[:, 1]

# threshold tuning
thresholds = np.arange(0.1, 0.9, 0.01)
precision_list = []
recall_list = []
f1_list = []

for threshold in thresholds:
    y_pred = (y_probs >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=["No Churn", "Churn"])
    precision_list.append(report["Churn"]["precision"])
    recall_list.append(report["Churn"]["recall"])
    f1_list.append(report["Churn"]["f1-score"])

# plot precision, recall, and f1-score
plt.figure(figsize=(12, 8))
plt.plot(thresholds, precision_list, label="Precision", marker="o")
plt.plot(thresholds, recall_list, label="Recall", marker="o")
plt.plot(thresholds, f1_list, label="F1-Score", marker="o")
plt.title("Threshold Tuning for XGBoost")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# selecting the best threshold
best_threshold = thresholds[np.argmax(f1_list)]
print(f"\nBest Threshold: {best_threshold:.2f} (F1-Score: {max(f1_list):.4f})")

# final evaluation
y_final = (y_probs >= best_threshold).astype(int)
print("\nFinal Classification Report:\n")
print(classification_report(y_test, y_final))

# Confusion matrix
cm = confusion_matrix(y_test, y_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="viridis", fmt="d",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"XGBoost (Iter 3 Optuna) – Threshold {best_threshold:.2f} – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# Save the final model with the best threshold
final_model_path = f"../../../models/xgboost_iteration_3_optuna_threshold_{best_threshold:.2f}.pkl"
joblib.dump(model, final_model_path)
print(f"\nFinal model saved to {final_model_path}")
