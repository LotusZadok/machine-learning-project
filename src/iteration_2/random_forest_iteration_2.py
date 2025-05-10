import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load the processed data
df = pd.read_csv('./data/processed/telco_customer_churn_encoded.csv')

# Refined features after correlation analysis
refined_features = [
    'PaymentMethod_Credit card (automatic)', 'SeniorCitizen', 
    'Contract_One year', 'StreamingMovies_Yes', 'Dependents_Yes', 
    'OnlineSecurity_Yes', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check', 'gender_Male', 'MonthlyCharges', 
    'OnlineBackup_Yes', 'InternetService_Fiber optic', 
    'MultipleLines_Yes', 'TechSupport_Yes', 'DeviceProtection_Yes', 
    'Contract_Two year', 'tenure', 'StreamingTV_Yes', 'PaperlessBilling_Yes'
]

X = df[refined_features]
y = df['Churn_Yes']

# Split the data before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("Class Distribution in Training Set:")
print(y_train_resampled.value_counts())

# Standardize the features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Revalidate with Random Forest
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best Model
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Evaluate on the test set
y_pred = best_model.predict(X_test)
print("\nTest Score:", best_model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Random Forest (Iteration 2 - Refined) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the best model
joblib.dump(best_model, './models/random_forest_iteration_2_refined.pkl')
