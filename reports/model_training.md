# Model Training - First Iteration Report

---

## Project Overview

This report summarizes the initial training phase for the **Telco Customer Churn** prediction project. The goal of this phase was to establish baseline performance for four different machine learning models:

1. **Logistic Regression**
2. **Random Forest**
3. **Support Vector Machine (SVM)**
4. **XGBoost (Gradient Boosting)**

---

## Data Preparation Recap

- **Numerical Scaling:** Applied **StandardScaler** to `tenure`, `MonthlyCharges`, and `TotalCharges`.
- **Class Balancing:** Used **SMOTE** to balance the target variable (`Churn_Yes`).
- **One-Hot Encoding:** Converted all categorical variables to numerical form.

---

## Model Performance Summary

| Model                  | CV Accuracy         | Test Accuracy | Precision (True) | Recall (True) | F1-Score (True) |
| ---------------------- | ------------------- | ------------- | ---------------- | ------------- | --------------- |
| Logistic Regression    | **0.8007 ± 0.0124** | **0.7933**    | **0.77**         | **0.84**      | **0.80**        |
| Random Forest          | **0.8380 ± 0.0078** | **0.8349**    | **0.82**         | **0.86**      | **0.84**        |
| Support Vector Machine | **0.7990 ± 0.0151** | **0.7899**    | **0.77**         | **0.83**      | **0.80**        |
| XGBoost                | **0.8266 ± 0.0104** | **0.8291**    | **0.81**         | **0.86**      | **0.83**        |

---

## Detailed Model Analysis

### Logistic Regression

- **Cross-Validation Accuracy:** 0.8007 ± 0.0124
- **Test Accuracy:** 0.7933
- **Confusion Matrix:**
  ```
  [[772 261]
   [166 867]]
  ```
- **Observations:**
  - Good generalization but slightly lower precision for predicting churn.
  - Strong recall for the churn class, making it effective for minimizing customer loss.

---

### Random Forest

- **Cross-Validation Accuracy:** 0.8380 ± 0.0078
- **Test Accuracy:** 0.8349
- **Confusion Matrix:**
  ```
  [[839 194]
   [147 886]]
  ```
- **Observations:**
  - Best overall performance.
  - Balanced precision and recall, indicating strong predictive power without overfitting.

---

### Support Vector Machine (SVM)

- **Cross-Validation Accuracy:** 0.7990 ± 0.0151
- **Test Accuracy:** 0.7899
- **Confusion Matrix:**
  ```
  [[774 259]
   [175 858]]
  ```
- **Observations:**
  - Slightly lower accuracy than the other models.
  - Higher sensitivity to data variations, as indicated by the higher cross-validation deviation.

---

### XGBoost

- **Cross-Validation Accuracy:** 0.8266 ± 0.0104
- **Test Accuracy:** 0.8291
- **Confusion Matrix:**
  ```
  [[826 207]
   [146 887]]
  ```
- **Observations:**
  - Strong performance, close to Random Forest.
  - Effective balance between precision and recall, with a slight trade-off in overall accuracy.

---

## General Observations

- **Best Initial Model:** **Random Forest** achieved the highest overall accuracy and F1-score, indicating it is the most promising model at this stage.
- **Consistent Performance:** All models showed consistent performance between training and testing, indicating minimal overfitting.
- **Precision vs. Recall Trade-Off:** While **Random Forest** and **XGBoost** balanced precision and recall well, **Logistic Regression** leaned towards higher recall, which may be preferable in customer retention scenarios.
- **Cross-Validation Variability:** SVM showed the most variability in cross-validation, suggesting a need for parameter tuning.

---

## Next Steps

- **Hyperparameter Tuning:** Use **GridSearchCV** or **Optuna** to optimize each model.
- **Feature Importance Analysis:** Investigate which features have the highest predictive power, especially for tree-based models.
- **Overfitting Analysis:** Add regularization for models like **Logistic Regression** and **SVM**.
