## Data Preparation

### Key Steps

#### (a) Data Selection

- All columns were selected for analysis except `customerID`, which is a unique identifier without predictive power.

#### (b) Data Cleaning

- Converted `TotalCharges` from object to float, removing 11 rows with missing values.
- Verified that no columns contain missing values after cleaning.
- Verified that there are no duplicate rows.

#### (c) New Features (Optional)

- Consider adding new features in future iterations, such as:
  - **Lifetime Value (LTV)** based on `tenure` and `MonthlyCharges`.
  - **Average Monthly Spend** calculated as `TotalCharges / tenure`.

#### (d) Data Transformations

- Applied one-hot encoding to all categorical variables to prepare for machine learning algorithms.
- Saved the cleaned and encoded data to `./data/processed/telco_customer_churn_encoded.csv`.

---

### Normalization of Numerical Features

To ensure that the models do not favor features with larger numerical ranges, the following columns were normalized using **StandardScaler**:

- **tenure**
- **MonthlyCharges**
- **TotalCharges**

Normalization helps stabilize the learning process and reduces the impact of large outliers.

---

### Balancing the `Churn` Field

The target variable **`Churn`** was found to be imbalanced, which can lead to biased model predictions. To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to create a balanced dataset.

- This approach generates synthetic samples for the minority class (`Churn_Yes`), effectively balancing the classes.
- This step helps improve the model's ability to detect both positive and negative cases of churn.

---

### Final Output

- The fully prepared and balanced dataset was saved to `./data/processed/telco_customer_churn_final.csv`.

---
