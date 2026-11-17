# Model Evaluation Report

Dataset: Kaggle Titanic training data. Hold-out split: 80% train / 20% test,
stratified, random_state=42. Preprocessing is fitted inside each pipeline
to avoid fitting imputers, encoders, or scalers on the test set.

## Hold-out test metrics

| model               |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:--------------------|-----------:|------------:|---------:|---------:|----------:|
| Logistic Regression |   0.826816 |    0.756757 | 0.811594 | 0.783217 |  0.871673 |
| Random Forest       |   0.810056 |    0.746479 | 0.768116 | 0.757143 |  0.846443 |
| Decision Tree       |   0.787709 |    0.691358 | 0.811594 | 0.746667 |  0.845191 |
| Tuned Decision Tree |   0.793296 |    0.690476 | 0.84058  | 0.75817  |  0.829974 |

## Selected model

**Logistic Regression** was selected by the highest hold-out ROC-AUC in this run.

## Interpretation and limitations

- Accuracy alone can hide class-specific errors; precision, recall, F1 and ROC-AUC are also reported.
- The dataset is historical and relatively small; results may not generalize to other populations or contexts.
- Some columns have missing values, and imputation can introduce uncertainty.
- Feature associations are not causal explanations.
- The train/test result is one split; cross-validation on the training partition provides an additional stability check.

## Reproducibility

Run `python -m src.train` from the repository root after placing Kaggle `train.csv` in `data/raw/`.
The selected fitted preprocessing + model pipeline is saved as `models/titanic_model.pkl`.