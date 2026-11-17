# Model Card — Titanic Survival Classifier

## Model details
- **Model type:** Logistic Regression
- **Artifact:** `models/titanic_model.pkl`
- **Task:** Binary classification of the `Survived` target
- **Selected model:** Logistic Regression, selected based on hold-out ROC-AUC
- **Framework:** Python and Scikit-learn

## Intended use
This model is intended for educational purposes and demonstrates a complete tabular machine learning workflow. It is not intended for real-world decisions about individuals.

## Data
- **Source:** Kaggle Titanic competition training dataset
- **File:** `data/raw/train.csv`
- **Dataset size:** 891 passenger records and 12 original columns
- **Target:** `Survived` (0 = did not survive, 1 = survived)
- **Missing values:** Age (177), Cabin (687), Embarked (2)
- **Preprocessing:** Numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding.
- **Feature engineering:** Title, FamilySize, IsAlone and CabinDeck.

## Evaluation

The selected Logistic Regression model was evaluated on a held-out test set.

| Metric | Test result |
|---|---:|
| Accuracy | 82.68% |
| Precision | 75.68% |
| Recall | 81.16% |
| F1-score | 78.32% |
| ROC-AUC | 87.17% |

### Cross-validation
Five-fold cross-validation results:

| Model | Mean accuracy | Standard deviation |
|---|---:|---:|
| Logistic Regression | 0.819 | 0.034 |
| Decision Tree | 0.789 | 0.026 |
| Random Forest | 0.832 | 0.033 |

The cross-validation accuracy and final hold-out metrics measure different evaluations and should not be directly compared as if they were the same test.

## Bias and fairness considerations
The dataset reflects historical passenger records and social conditions. Outcomes may be associated with passenger class, sex, age and other attributes. These patterns may encode historical inequities and should not be interpreted as fair or causal rules.

This small dataset does not support a comprehensive fairness audit. Subgroup performance, for example by sex and passenger class, should be evaluated before making claims about differences in model performance.

## Limitations
- Small historical dataset with limited external validity.
- Missing data and imputation introduce uncertainty.
- A single hold-out split may produce variable performance estimates.
- Historical patterns may reflect social inequalities.
- Model predictions and feature importance do not establish causality.
- Not intended for deployment or consequential decisions.

## Reproduction
1. Install dependencies from `requirements.txt`.
2. Download the Titanic training dataset from Kaggle.
3. Place `train.csv` in `data/raw/`.
4. Run the training script:

   `python -m src.train`

5. Run the tests:

   `python -m pytest -v`

The training script saves the selected model to `models/titanic_model.pkl` and generates evaluation reports.

## Prediction
Run the prediction script using the project's prepared sample input:

`python -m src.predict`

## Model artifact
The trained model is saved as `models/titanic_model.pkl`. The artifact should be used with the same feature engineering and preprocessing workflow used during training.
