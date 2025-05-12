# Selection and Conclusions

## Model Evaluation Summary

After multiple iterations, including hyperparameter tuning and feature refinement, the following models were developed and evaluated:

### 1. Logistic Regression (Iteration 2)

- **Best Parameters:** `{'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}`
- **Test Accuracy:** 0.7706
- **F1-Score:** 0.7706
- **Confusion Matrix:** Significant imbalance in predicting churn and non-churn customers.
- **Conclusion:** This model showed the weakest performance, struggling to generalize on both classes despite attempts to reduce multicollinearity.

### 2. Random Forest (Iteration 2)

- **Best Parameters:** `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 500}`
- **Test Accuracy:** 0.8345
- **F1-Score:** 0.8345
- **Confusion Matrix:** Balanced performance across both classes, with reduced false negatives.
- **Conclusion:** Effective at capturing complex relationships but prone to overfitting without regularization.

### 3. SVM (Iteration 2)

- **Best Parameters:** `{'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}`
- **Test Accuracy:** 0.8146
- **F1-Score:** 0.8146
- **Confusion Matrix:** Moderate trade-off between precision and recall, showing stability after tuning.
- **Conclusion:** Effective for linearly separable data but less flexible than tree-based methods.

### 4. XGBoost (Iteration 3 - Optuna)

- **Best Parameters:** `{'subsample': 0.7, 'reg_lambda': 2, 'reg_alpha': 0, 'n_estimators': 600, 'min_child_weight': 1, 'max_depth': 9, 'learning_rate': 0.042, 'gamma': 0.4, 'colsample_bytree': 1.0}`
- **Test AUC:** 0.9117
- **F1-Score:** 0.8401 (with threshold adjustment at 0.41)
- **Confusion Matrix:** Balanced results, effectively managing false positives and negatives after threshold tuning.
- **Conclusion:** Demonstrated the highest overall performance, making it the best candidate for this project.

---

## Final Model Selection

Based on these evaluations, **XGBoost** stands out as the most robust model due to its superior performance metrics:

- High AUC (0.9117), indicating strong generalization.
- Optimal trade-off between precision and recall with threshold adjustment.
- Stability across different hyperparameter configurations, confirmed via Optuna tuning.

This model not only captures the complexity of the churn prediction task but also maintains a balanced error rate, making it the most suitable choice for this dataset.

---

## Conclusion

With the current model achieving a balanced performance across key metrics, the project can be considered successful within the scope of this task.
