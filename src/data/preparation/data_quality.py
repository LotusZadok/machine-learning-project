import pandas as pd

df = pd.read_csv('../../data/raw/telco_customer_churn.csv')

# null values verification
missing_values = df.isnull().sum()
print("Missing Values per Column:")
print(missing_values[missing_values > 0])

# duplicates verification
duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicates: {duplicates}")

# inconsistent data types verification
print("\nData Types:")
print(df.dtypes)

# unique values verification
print("\nUnique Values per Categorical Column:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].nunique()} unique values")
