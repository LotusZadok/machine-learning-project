## Data Understanding

### Data Sources

- **Dataset Name:** Telco Customer Churn
- **Source:** Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Access Method:** Downloaded as CSV file
- **Description:** This dataset contains customer data from a telecommunications company. It includes demographic information, services subscribed, billing data, and churn status. The primary goal is to predict whether a customer will leave the company.
- **File Name:** telco_customer_churn.csv
- **Total Records:** 7,043 rows, 21 columns

### Key Variables:

- **customerID:** Unique customer identifier.
- **gender:** Gender of the customer (Male/Female).
- **SeniorCitizen:** Whether the customer is a senior citizen (0, 1).
- **tenure:** Months the customer has been with the company.
- **MonthlyCharges:** Amount charged monthly.
- **TotalCharges:** Total amount billed to the customer.
- **Churn:** Whether the customer left the service (Yes/No).

## Data Overview

The **Telco Customer Churn** dataset contains 7,043 rows and 21 columns. It includes customer demographics, service subscriptions, billing information, and churn status, providing a comprehensive view of customer behavior and financial interactions.

---

### Data Types

| Column Name      | Data Type | Description                                                                     |
| ---------------- | --------- | ------------------------------------------------------------------------------- |
| customerID       | Object    | Unique customer identifier.                                                     |
| gender           | Object    | Gender of the customer (Male/Female).                                           |
| SeniorCitizen    | Integer   | Indicates if the customer is a senior citizen (0: No, 1: Yes).                  |
| Partner          | Object    | Whether the customer has a partner (Yes/No).                                    |
| Dependents       | Object    | Whether the customer has dependents (Yes/No).                                   |
| tenure           | Integer   | Number of months the customer has been with the company.                        |
| PhoneService     | Object    | Whether the customer has a phone service (Yes/No).                              |
| MultipleLines    | Object    | Whether the customer has multiple phone lines (Yes/No/No phone service).        |
| InternetService  | Object    | Type of internet service (DSL, Fiber optic, No).                                |
| OnlineSecurity   | Object    | Whether the customer has online security (Yes/No/No internet service).          |
| OnlineBackup     | Object    | Whether the customer has online backup (Yes/No/No internet service).            |
| DeviceProtection | Object    | Whether the customer has device protection (Yes/No/No internet service).        |
| TechSupport      | Object    | Whether the customer has tech support (Yes/No/No internet service).             |
| StreamingTV      | Object    | Whether the customer has streaming TV service (Yes/No/No internet service).     |
| StreamingMovies  | Object    | Whether the customer has streaming movies service (Yes/No/No internet service). |
| Contract         | Object    | Type of contract (Month-to-month, One year, Two year).                          |
| PaperlessBilling | Object    | Whether the customer uses paperless billing (Yes/No).                           |
| PaymentMethod    | Object    | Payment method (Electronic check, Mailed check, Bank transfer, Credit card).    |
| MonthlyCharges   | Float     | Amount charged to the customer monthly.                                         |
| TotalCharges     | Object    | Total amount billed to the customer. Requires conversion to numeric.            |
| Churn            | Object    | Whether the customer left the service (Yes/No).                                 |

---

### Descriptive Statistics

| Metric    | SeniorCitizen | tenure | MonthlyCharges |
| --------- | ------------- | ------ | -------------- |
| **Count** | 7043          | 7043   | 7043           |
| **Mean**  | 0.162         | 32.37  | 64.76          |
| **Std**   | 0.369         | 24.56  | 30.09          |
| **Min**   | 0             | 0      | 18.25          |
| **25%**   | 0             | 9      | 35.50          |
| **50%**   | 0             | 29     | 70.35          |
| **75%**   | 0             | 55     | 89.85          |
| **Max**   | 1             | 72     | 118.75         |

---

### Initial Insights

- The dataset has a significant imbalance in the `Churn` variable, with more customers labeled as "No" than "Yes" (as seen in the first bar chart).
- **MonthlyCharges** has a wide distribution, ranging from approximately 18 to 120, with a peak around 20.
- The **TotalCharges** column is stored as an object and requires conversion to a numeric format for proper analysis.
- **SeniorCitizen** is a binary variable with a mean of 0.16, indicating a small proportion of senior customers.
- There are correlations between `tenure`, `MonthlyCharges`, and `SeniorCitizen`, but the overall correlations are weak, as indicated by the heatmap.

## Data Quality Check

### Missing Values

- No missing values detected in any column.

---

### Duplicate Records

- **Number of Duplicates:** 0

---

### Data Types

| Column Name      | Data Type |
| ---------------- | --------- |
| customerID       | Object    |
| gender           | Object    |
| SeniorCitizen    | Integer   |
| Partner          | Object    |
| Dependents       | Object    |
| tenure           | Integer   |
| PhoneService     | Object    |
| MultipleLines    | Object    |
| InternetService  | Object    |
| OnlineSecurity   | Object    |
| OnlineBackup     | Object    |
| DeviceProtection | Object    |
| TechSupport      | Object    |
| StreamingTV      | Object    |
| StreamingMovies  | Object    |
| Contract         | Object    |
| PaperlessBilling | Object    |
| PaymentMethod    | Object    |
| MonthlyCharges   | Float     |
| TotalCharges     | Object    |
| Churn            | Object    |

---

### Unique Values per Categorical Column

| Column Name      | Unique Values |
| ---------------- | ------------- |
| customerID       | 7043          |
| gender           | 2             |
| Partner          | 2             |
| Dependents       | 2             |
| PhoneService     | 2             |
| MultipleLines    | 3             |
| InternetService  | 3             |
| OnlineSecurity   | 3             |
| OnlineBackup     | 3             |
| DeviceProtection | 3             |
| TechSupport      | 3             |
| StreamingTV      | 3             |
| StreamingMovies  | 3             |
| Contract         | 3             |
| PaperlessBilling | 2             |
| PaymentMethod    | 4             |
| TotalCharges     | 6531          |
| Churn            | 2             |

---

### Key Observations

- The `customerID` column has 7,043 unique values, indicating no duplicate customer records.
- The `TotalCharges` column has 6,531 unique values, which is fewer than the total number of rows, suggesting potential issues with data consistency.
- Many categorical columns (e.g., `MultipleLines`, `InternetService`, `Contract`) have a limited number of unique values, suitable for one-hot encoding or label encoding in the data preparation phase.
- The `TotalCharges` column is incorrectly stored as an object and requires conversion to numeric for analysis.
