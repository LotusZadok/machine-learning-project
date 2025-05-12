import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../../data/raw/telco_customer_churn.csv')

# totalcharges to numeric conversion

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_total_charges = df['TotalCharges'].isna().sum()
print(f"Missing TotalCharges: {missing_total_charges}")

df.dropna(subset=['TotalCharges'], inplace=True)

print(df['TotalCharges'].describe())

# missing values and duplicates verification

print("Missing Values per Column:")
print(df.isna().sum())

print(f"\nNumber of Duplicates: {df.duplicated().sum()}")

# categorical variables conversion

categorical_columns = [
    'gender','Partner','Dependents','PhoneService','MultipleLines',
    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies','Contract',
    'PaperlessBilling','PaymentMethod','Churn'
]

df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# normalization of numerical variables with standard scaler

df_encoded.drop(columns=['customerID'], inplace=True)
numerical_columns = [
    'SeniorCitizen','tenure','MonthlyCharges','TotalCharges'
]
scaler = StandardScaler()
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

x = df_encoded.drop(columns=['Churn_Yes'], axis=1)
y = df_encoded['Churn_Yes']

smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)

df_resampled = pd.concat([pd.DataFrame(x_resampled, columns=x.columns), pd.DataFrame(y_resampled, columns=['Churn_Yes'])], axis=1)
print(df_resampled['Churn_Yes'].value_counts())

df_resampled.to_csv('../../data/processed/telco_customer_churn_encoded.csv', index=False)



print("Data preparation completed and saved to ./data/processed/telco_customer_churn_encoded.csv")
# This code reads a CSV file, processes the data by converting the 'TotalCharges' column to numeric,