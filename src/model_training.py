import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# Load the processed data
df = pd.read_csv('./data/processed/telco_customer_churn_encoded.csv')

# Separate features and target variable
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

# Train and evaluate each model
for name, model in models.items():
    # Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n{name} Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"\n{name} Test Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, f'./models/{name.replace(" ", "_").lower()}_model.pkl')
    print(f"Model saved: ./models/{name.replace(' ', '_').lower()}_model.pkl")
