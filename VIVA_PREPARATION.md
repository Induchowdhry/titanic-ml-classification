# Project Viva Preparation — Titanic ML

Use these as revision notes. Explain in your own words and be ready to point to the relevant code.

1. **What is the project?** A binary classification project that estimates the `Survived` label from Titanic passenger attributes.
2. **Why Titanic?** It is a beginner-friendly, labelled tabular dataset with numeric and categorical features and missing values.
3. **What is EDA?** Exploratory Data Analysis: inspecting shape, types, distributions, missingness, relationships and class balance before modelling.
4. **Why handle missing values?** Most Scikit-Learn estimators cannot directly process missing values. Median is robust for numeric columns; mode is a simple categorical strategy.
5. **What is feature engineering?** Creating useful input variables from existing data, such as family size and extracted title.
6. **What is one-hot encoding?** Representing categories as binary columns without imposing an artificial numeric order.
7. **Why scale features?** Scaling puts numeric features on comparable ranges; it is particularly useful for Logistic Regression and distance/gradient-based methods. Tree models generally do not require scaling, but a shared pipeline keeps preprocessing consistent here.
8. **What is data leakage?** Using information from the test set or future/target information during training. Fitting preprocessing only inside the training pipeline helps prevent it.
9. **Why stratified splitting?** It keeps approximately the same target-class proportions in train and test sets.
10. **Logistic Regression?** A linear classifier that models class probability using the logistic function.
11. **Decision Tree?** A model that recursively splits data using feature-based rules.
12. **Random Forest?** An ensemble that combines many decision trees to reduce variance compared with a single tree.
13. **Overfitting?** When a model learns training-specific patterns and performs worse on unseen data. Limiting tree depth and minimum leaf size can help.
14. **Accuracy?** Fraction of predictions that are correct.
15. **Precision?** Of predicted positive cases, the fraction that are truly positive.
16. **Recall?** Of actual positive cases, the fraction correctly identified.
17. **F1-score?** Harmonic mean of precision and recall.
18. **ROC-AUC?** Measures how well scores rank positive examples above negative examples across classification thresholds.
19. **Confusion matrix?** Counts true negatives, false positives, false negatives and true positives.
20. **Cross-validation?** Repeats training/validation across folds to estimate performance more robustly.
21. **GridSearchCV?** Evaluates specified hyperparameter combinations using cross-validation and selects the best scoring combination.
22. **Why save the full pipeline?** It preserves preprocessing and the estimator together, so inference uses the same transformations as training.
23. **Why joblib?** It efficiently serializes Python objects such as fitted Scikit-Learn pipelines.
24. **Limitations?** Historical small dataset, missing values, possible bias, limited generalization and no causal interpretation.
25. **What would you improve?** More robust external validation, subgroup error analysis, calibration, explainability and better documentation of data provenance.

## Be prepared to demonstrate
- Run the notebook from top to bottom.
- Show the missing-value summary and EDA charts.
- Explain the engineered features.
- Show the pipeline and train/test split.
- Compare the actual metrics from your generated report.
- Load the saved model and run the sample prediction.
- Show the GitHub Actions test result.
