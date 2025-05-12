# Hyperparameter Tuning - First Iteration Report

---

## Objective

The goal of this phase was to optimize the hyperparameters of the four initial models to improve their performance in terms of accuracy, precision, recall, and F1-score. This step aims to reduce overfitting and select the most robust models for the final evaluation.

---

## Models Included in Hyperparameter Tuning

1. **Logistic Regression**
2. **Random Forest**
3. **Support Vector Machine (SVM)**
4. **XGBoost (Gradient Boosting)**

---

## Hyperparameters Tuned

| Model               | Tuned Hyperparameters                                                         |
| ------------------- | ----------------------------------------------------------------------------- |
| Logistic Regression | `C`, `penalty`, `solver`                                                      |
| Random Forest       | `n_estimators`, `max_depth`, `max_features`, `min_samples_split`              |
| SVM                 | `C`, `kernel`, `gamma`                                                        |
| XGBoost             | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |

---

## Tuning Results

### Logistic Regression

- **Best Hyperparameters:**
  {
  'C': 1,
  'penalty': 'l1',
  'solver': 'saga'
  }

- **Regularization Strength (`C`)**: 1 (Moderate regularization)
- **Penalty**: L1 (Lasso, promotes sparsity in coefficients)
- **Solver**: SAGA (Optimized for L1 penalty with large datasets)

---

## Performance Metrics

| Metric           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **False**        | 0.83      | 0.75   | 0.79     | 1033    |
| **True**         | 0.77      | 0.84   | 0.81     | 1033    |
| **Accuracy**     | 0.80      | -      | -        | 2066    |
| **Macro Avg**    | 0.80      | 0.80   | 0.80     | 2066    |
| **Weighted Avg** | 0.80      | 0.80   | 0.80     | 2066    |

---

## Confusion Matrix

[[778 255]
[163 870]]

- **True Negatives**: 778
- **False Positives**: 255
- **False Negatives**: 163
- **True Positives**: 870

---

## Observations

- **Overall Accuracy**: The tuned model achieved an accuracy of **0.80**, slightly lower than the best cross-validation score (**0.8047**), indicating minor overfitting.
- **Precision and Recall Balance**:
  - **Precision for No Churn**: 0.83
  - **Recall for Churn**: 0.84
  - Balanced performance, making it effective at capturing true churn cases without significantly increasing false positives.
- **Regularization Impact**:
  - The **L1 regularization** (`penalty='l1'`) promotes sparse coefficients, reducing model complexity and potentially improving interpretability.
- **Solver Efficiency**:
  - The **SAGA** solver is optimized for large datasets with L1 regularization, making it more suitable for this dataset.

---

## Convergence Warning

ConvergenceWarning: The max*iter was reached which means the coef* did not converge

- **Issue**: The model did not fully converge within the specified **`max_iter=1000`**.
- **Impact**: The coefficients may not have reached optimal values, potentially affecting the overall stability of the model.
- **Suggested Solutions**:
  - Increase the **`max_iter`** parameter to 2000 or 5000 to ensure full convergence.
  - Consider adjusting the learning rate or regularization strength if the coefficients remain unstable.

---

## Next Steps

- **Feature Importance Analysis**: Identify the most impactful features contributing to churn prediction.
- **Hyperparameter Refinement**: Expand the hyperparameter grid with additional solvers (`liblinear`, `newton-cg`) and higher `C` values.
- **Overfitting Check**: Perform additional validation to ensure the model generalizes well to new data.

---

### Random Forest

- **Best Hyperparameters:**
  {
  'max_depth': None,
  'max_features': 'log2',
  'min_samples_split': 2,
  'n_estimators': 200
  }

## Performance Metrics

| Metric           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **False**        | 0.86      | 0.81   | 0.83     | 1033    |
| **True**         | 0.82      | 0.86   | 0.84     | 1033    |
| **Accuracy**     | 0.84      | -      | -        | 2066    |
| **Macro Avg**    | 0.84      | 0.84   | 0.84     | 2066    |
| **Weighted Avg** | 0.84      | 0.84   | 0.84     | 2066    |

---

## Confusion Matrix

[[841 192]
[141 892]]

- **True Negatives**: 841
- **False Positives**: 192
- **False Negatives**: 141
- **True Positives**: 892

---

## Observations

- **Overall Accuracy**: The tuned model achieved an accuracy of **0.84**, closely matching the best cross-validation score (**0.8408**), indicating minimal overfitting.
- **Precision and Recall Balance**:
  - **Precision for No Churn**: 0.86
  - **Recall for Churn**: 0.86
  - Excellent balance between precision and recall, making it highly effective at correctly identifying churn cases.
- **Feature Utilization**:
  - Using **log2** for `max_features` reduces the risk of overfitting by limiting the number of features considered at each split, improving generalization.
- **Deep Trees**:
  - Allowing trees to grow without depth limits (**`max_depth=None`**) captures more complex relationships in the data, potentially improving predictive power.
- **High Recall for Churn**:
  - The high recall for the churn class (0.86) indicates that the model captures the majority of churn cases, which is crucial for minimizing customer loss.

---

## Next Steps

- **Feature Importance Analysis**: Identify the most impactful features driving model decisions.
- **Overfitting Check**: Monitor the model for potential overfitting given the unrestricted tree depth.
- **Hyperparameter Refinement**: Consider adding regularization (e.g., `min_samples_leaf`) to reduce the risk of overfitting further.

### Support Vector Machine (SVM)

---

## Best Hyperparameters

{
'C': 10,
'gamma': 'scale',
'kernel': 'rbf'
}

- **Regularization Parameter (`C`)**: 10 (Higher regularization, allowing for more complex decision boundaries)
- **Gamma**: scale (Automatically adjusts the kernel coefficient based on the number of features)
- **Kernel**: rbf (Radial Basis Function, effective for capturing non-linear relationships)

---

## Performance Metrics

| Metric           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **False**        | 0.84      | 0.78   | 0.81     | 1033    |
| **True**         | 0.80      | 0.85   | 0.83     | 1033    |
| **Accuracy**     | 0.82      | -      | -        | 2066    |
| **Macro Avg**    | 0.82      | 0.82   | 0.82     | 2066    |
| **Weighted Avg** | 0.82      | 0.82   | 0.82     | 2066    |

---

## Confusion Matrix

[[810 223]
[150 883]]

- **True Negatives**: 810
- **False Positives**: 223
- **False Negatives**: 150
- **True Positives**: 883

---

## Observations

- **Overall Accuracy**: The tuned model achieved an accuracy of **0.82**, close to the best cross-validation score (**0.817**), indicating good generalization.
- **Precision and Recall Balance**:
  - **Precision for No Churn**: 0.84
  - **Recall for Churn**: 0.85
  - The model demonstrates strong recall for identifying churn cases, which is crucial for minimizing customer loss.
- **Regularization Impact**:
  - The high **`C`** value (10) allows the model to fit a more complex decision boundary, potentially capturing more subtle patterns in the data.
- **Kernel Selection**:
  - The **RBF kernel** is effective at capturing non-linear relationships, which likely contributed to the model's improved recall.
- **Gamma Parameter**:
  - Using **`gamma='scale'`** ensures that the influence of each feature is appropriately scaled, reducing the risk of overfitting.

---

## Next Steps

- **Hyperparameter Refinement**: Consider exploring other kernels (e.g., `poly`, `sigmoid`) to further optimize performance.
- **Overfitting Check**: Validate the model on an independent test set to confirm generalization.
- **Feature Scaling Verification**: Ensure all input features are properly scaled, as SVMs are sensitive to feature magnitudes.

### XGBoost

---

## Best Hyperparameters

{
'colsample_bytree': 1.0,
'learning_rate': 0.2,
'max_depth': 7,
'n_estimators': 200,
'subsample': 0.8
}

- **Column Subsampling (`colsample_bytree`)**: 1.0 (Uses all features for each tree)
- **Learning Rate (`learning_rate`)**: 0.2 (Moderately high, faster convergence but risk of overfitting)
- **Maximum Depth (`max_depth`)**: 7 (Moderate tree depth, controls model complexity)
- **Number of Estimators (`n_estimators`)**: 200 (Number of boosting rounds)
- **Subsample**: 0.8 (Uses 80% of the data for each boosting round, reducing overfitting)

---

## Performance Metrics

| Metric           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **False**        | 0.84      | 0.81   | 0.82     | 1033    |
| **True**         | 0.82      | 0.85   | 0.83     | 1033    |
| **Accuracy**     | 0.83      | -      | -        | 2066    |
| **Macro Avg**    | 0.83      | 0.83   | 0.83     | 2066    |
| **Weighted Avg** | 0.83      | 0.83   | 0.83     | 2066    |

---

## Confusion Matrix

[[835 198]
[160 873]]

- **True Negatives**: 835
- **False Positives**: 198
- **False Negatives**: 160
- **True Positives**: 873

---

## Observations

- **Overall Accuracy**: The tuned model achieved an accuracy of **0.83**, closely aligning with the best cross-validation score (**0.834**), indicating effective generalization.
- **Precision and Recall Balance**:
  - **Precision for No Churn**: 0.84
  - **Recall for Churn**: 0.85
  - Balanced precision and recall, making the model effective at capturing both churn and non-churn cases.
- **Learning Rate Impact**:
  - The relatively high **learning rate (0.2)** allows the model to converge faster, but at the risk of overfitting.
- **Subsampling Strategy**:
  - Using **0.8** subsampling helps reduce overfitting by training each tree on a random subset of the data.
- **Tree Depth**:
  - The maximum depth of 7 strikes a balance between model complexity and generalization, capturing non-linear relationships without excessive overfitting.

---

## Next Steps

- **Feature Importance Analysis**: Identify the most influential features driving model predictions.
- **Learning Rate Refinement**: Consider reducing the learning rate and increasing the number of estimators for potentially better performance.
- **Regularization Tuning**: Add regularization parameters like **`alpha`** and **`lambda`** to reduce overfitting further.
