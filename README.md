# Titanic Survival Classification

A beginner-friendly end-to-end machine-learning project that predicts whether a Titanic passenger survived using tabular passenger data. It demonstrates exploratory data analysis, feature engineering, preprocessing, model training, evaluation, model persistence and reproducible testing.

## Project objectives

- Explore and understand a real-world tabular dataset using NumPy, Pandas, Matplotlib and Seaborn.
- Handle missing numerical and categorical values.
- Engineer features such as passenger title, family size and whether a passenger travelled alone.
- Build leakage-aware Scikit-Learn pipelines.
- Compare Logistic Regression, Decision Tree and Random Forest classifiers.
- Use stratified splitting, 5-fold cross-validation and GridSearchCV.
- Evaluate models using accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC.
- Save the selected complete pipeline with joblib and reuse it for predictions.

## Dataset

Use the official Kaggle Titanic competition dataset: https://www.kaggle.com/competitions/titanic/data

Download **train.csv** and place it at:

```text
data/raw/train.csv
```

The training file must contain the `Survived` target column. The dataset is not included in this repository because it is distributed by Kaggle and may require a Kaggle account/acceptance of its competition rules. Do not commit private Kaggle credentials or tokens.

## Repository structure

```text
titanic-ml-classification/
├── .github/workflows/ci.yml
├── data/raw/                 # Place Kaggle train.csv here
├── models/                   # Saved model is generated here
├── notebooks/
│   └── 01_titanic_eda_and_modeling.ipynb
├── reports/                  # Metrics, plots and evaluation report generated here
├── src/
│   ├── features.py
│   ├── predict.py
│   └── train.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

1. Download Kaggle `train.csv` and place it in `data/raw/train.csv`.
2. Start Jupyter:

```bash
jupyter notebook
```

3. Open `notebooks/01_titanic_eda_and_modeling.ipynb` and run all cells from top to bottom.
4. Run the reproducible training script:

```bash
python -m src.train
```

5. Run automated tests:

```bash
pytest -q
```

6. Test the saved model:

```bash
python -m src.predict
```

## Methodology

### EDA
The notebook checks dataset dimensions, data types, descriptive statistics, missing values, target balance, categorical distributions, numerical distributions and feature correlations. It includes histograms, box plots, scatter plots, a pair plot and a correlation heatmap.

### Feature engineering
- `Title`: extracts passenger title from the name.
- `FamilySize`: combines siblings/spouses and parents/children, plus the passenger.
- `IsAlone`: indicates whether family size equals one.
- `CabinDeck`: extracts the first cabin character when available.

### Preprocessing
- Numeric columns: median imputation followed by StandardScaler.
- Categorical columns: most-frequent imputation followed by OneHotEncoder.
- Preprocessing is kept inside Scikit-Learn Pipelines. It is fitted only on training folds, helping prevent data leakage.

### Models
- Logistic Regression: linear classification baseline.
- Decision Tree: interpretable non-linear model; depth and leaf size are controlled.
- Random Forest: ensemble of decision trees.
- GridSearchCV tunes Decision Tree depth and minimum leaf size using 5-fold stratified cross-validation.

### Evaluation
A stratified 80/20 train/test split with `random_state=42` is used. Metrics are calculated on the held-out test set. The selected model is the model with the highest hold-out ROC-AUC in that run. See `reports/EVALUATION_REPORT.md` after training.

## Outputs generated

- `models/titanic_model.pkl` — selected fitted preprocessing + model pipeline.
- `models/best_model_name.txt` — selected model name.
- `reports/model_metrics.csv` — comparison metrics.
- `reports/evaluation_plots.png` — ROC curves and selected-model confusion matrix.
- `reports/EVALUATION_REPORT.md` — generated evaluation report.

These files are generated after training and are excluded from Git by default. If your internship reviewer explicitly requires the model artifact or report in GitHub, review the generated files and selectively add them.

## Example prediction

After training:

```bash
python -m src.predict
```

The example predicts for two illustrative passenger records. These are demonstration inputs, not actual passenger records or verified outcomes.

## Testing and continuous integration

Run `pytest -q` locally. GitHub Actions runs the same test suite on pushes and pull requests to `main`/`master`. A green CI run is evidence that the automated tests passed in that workflow; check the Actions tab for the actual run.

## Limitations and responsible use

This is an educational exercise using a historical dataset. The model is not suitable for real-world decisions about people. The dataset is small, historical and not representative of modern populations. Missing-value imputation, feature selection and historical social patterns can affect results. Predictive associations do not establish causation.

## Reproducibility notes

The split and model random states are fixed at 42. Actual metrics depend on the dataset file and installed library versions; use the generated report from your own run rather than copying example scores from elsewhere.

## Author

Add your name, internship details and GitHub profile here before submission.
