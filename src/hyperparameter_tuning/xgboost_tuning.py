import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load the data
df = pd.read_csv('../../data/processed/telco_customer_churn_encoded.csv')
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the model and hyperparameter grid
model = XGBClassifier(random_state=42, eval_metric='logloss')
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Perform GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Save the best model
joblib.dump(grid_search.best_estimator_, '../../models/xgboost_tuned.pkl')

# Test the best model
y_pred = grid_search.best_estimator_.predict(X_test)

print("\nBest Parameters:", grid_search.best_params_)
print("\nBest CV Score:", grid_search.best_score_)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
