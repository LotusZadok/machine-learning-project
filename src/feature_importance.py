import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load the processed data
df = pd.read_csv('./data/processed/telco_customer_churn_encoded.csv')
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

# Load the trained models
models = {
    "Random Forest": joblib.load('./models/random_forest_tuned.pkl'),
    "XGBoost": joblib.load('./models/xgboost_tuned.pkl')
}

plt.figure(figsize=(20, 14))

for i, (name, model) in enumerate(models.items(), 1):
    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        # Plot feature importances
        plt.subplot(len(models), 1, i)
        sns.barplot(x="Importance", y="Feature", data=importance_df.head(20), palette="viridis")
        plt.title(f"Top 20 Feature Importances - {name}", fontsize=24)
        plt.xlabel("Importance", fontsize=20)
        plt.ylabel("Feature", fontsize=20)

plt.tight_layout()
plt.show()
