import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data/processed/telco_customer_churn_encoded.csv')

refined_features = [
    'PaymentMethod_Credit card (automatic)', 'SeniorCitizen', 
    'Contract_One year', 'StreamingMovies_Yes', 'Dependents_Yes', 
    'OnlineSecurity_Yes', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check', 'gender_Male', 'MonthlyCharges', 
    'OnlineBackup_Yes', 'InternetService_Fiber optic', 
    'MultipleLines_Yes', 'TechSupport_Yes', 'DeviceProtection_Yes', 
    'Contract_Two year', 'tenure', 'StreamingTV_Yes', 'PaperlessBilling_Yes'
]

# removing refined features from the original dataset
X = df[refined_features]
y = df['Churn_Yes']

# multicollinearity check
plt.figure(figsize=(20, 15))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='viridis')
plt.title("Correlation Matrix - Refined Features")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# apply smote to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("Class Distribution in Training Set:")
print(y_train_resampled.value_counts())

# standardize the features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# picking the best model
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# evalute the model on the test set
y_pred = best_model.predict(X_test)
print("\nTest Score:", best_model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Logistic Regression (Iteration 2 - Refined) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the best model
joblib.dump(best_model, './models/logistic_regression_iteration_2_refined.pkl')
