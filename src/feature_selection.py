import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load
df = pd.read_csv('./data/processed/telco_customer_churn_encoded.csv')
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

# trained models
rf_model = joblib.load('./models/random_forest_tuned.pkl')
xgb_model = joblib.load('./models/xgboost_tuned.pkl')

# feature importances
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# set tjhreshold for feature importance
threshold = 0.01
important_features_rf = rf_importances[rf_importances > threshold].index
important_features_xgb = xgb_importances[xgb_importances > threshold].index

# get the features that are important in both models
selected_features = list(set(important_features_rf).intersection(set(important_features_xgb)))
print(f"Selected Features ({len(selected_features)}):", selected_features)
# Retrain models with selected features
X_reduced = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)

rf_retrained = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
xgb_retrained = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.2, subsample=0.8, random_state=42)

rf_retrained.fit(X_train, y_train)
xgb_retrained.fit(X_train, y_train)

models = {"Random Forest (Reduced)": rf_retrained, "XGBoost (Reduced)": xgb_retrained}

plt.figure(figsize=(20, 10))

for i, (name, model) in enumerate(models.items(), 1):
    y_pred = model.predict(X_test)
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))

    plt.subplot(1, 2, i)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()
