import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load the processed data
df = pd.read_csv('../../data/processed/telco_customer_churn_final.csv')
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

# define models and their parameters for validation curves
models_params = {
    "Logistic Regression": {
        "model": LogisticRegression(penalty='l1', solver='saga', max_iter=5000),
        "param_name": "C",
        "param_range": np.logspace(-2, 2, 5)
    },
    "Random Forest": {
        "model": RandomForestClassifier(max_depth=None, max_features='log2', min_samples_split=2, n_estimators=200),
        "param_name": "n_estimators",
        "param_range": [50, 100, 150, 200, 250, 300]
    },
    "SVM": {
        "model": SVC(kernel='rbf', gamma='scale'),
        "param_name": "C",
        "param_range": [0.1, 1, 10, 100, 1000]
    },
    "XGBoost": {
        "model": XGBClassifier(colsample_bytree=1.0, learning_rate=0.2, max_depth=7, subsample=0.8, eval_metric='logloss'),
        "param_name": "n_estimators",
        "param_range": [50, 100, 150, 200, 250, 300]
    }
}

plt.figure(figsize=(18, 12))

for i, (name, config) in enumerate(models_params.items(), 1):
    model = config["model"]
    param_name = config["param_name"]
    param_range = config["param_range"]
    
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=5, n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.subplot(2, 2, i)
    plt.plot(param_range, train_mean, color="blue", marker='o', label="Training score")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    
    plt.plot(param_range, test_mean, color="green", marker='o', label="Cross-validation score")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="green")
    
    plt.title(f"Validation Curve - {name}")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.xscale("log" if name == "Logistic Regression" or name == "SVM" else "linear")

plt.tight_layout()
plt.show()
