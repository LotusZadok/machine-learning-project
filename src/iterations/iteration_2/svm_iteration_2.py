import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../../../data/processed/telco_customer_churn_encoded.csv')

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

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# smote to balance the classes
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
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# grid search for hyperparameter tuning

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# final model evaluation

best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

y_pred = best_model.predict(X_test)
print("\nTest Score:", best_model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("SVM (Iteration 2 - Refined) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

joblib.dump(best_model, '../../../models/svm_iteration_2_refined.pkl')
