import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed data
df = pd.read_csv('./data/processed/telco_customer_churn_encoded.csv')
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

# Selected features (from previous step)
selected_features = [
    'PaymentMethod_Credit card (automatic)', 'SeniorCitizen', 'Contract_One year', 
    'StreamingMovies_Yes', 'Dependents_Yes', 'OnlineSecurity_Yes', 
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'gender_Male', 
    'TotalCharges', 'OnlineBackup_Yes', 'InternetService_Fiber optic', 
    'MultipleLines_Yes', 'TechSupport_Yes', 'DeviceProtection_Yes', 
    'MonthlyCharges', 'Contract_Two year', 'tenure', 'StreamingTV_Yes', 
    'PaperlessBilling_Yes'
]

X_reduced = X[selected_features]

# Split the reduced data
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    "Random Forest (Reduced)": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost (Reduced)": XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.2, subsample=0.8, random_state=42),
    "Logistic Regression (Reduced)": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (Reduced)": SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
}

# Train and evaluate models
plt.figure(figsize=(24, 20))

for i, (name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.subplot(2, 2, i)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()
