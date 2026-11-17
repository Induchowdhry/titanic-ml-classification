
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def test_preprocessing_and_classifier_fit_on_small_fixture():
    df = pd.DataFrame({
        "Age": [22, 38, None, 35, 28, 19, 45, 31, 26, 40],
        "Fare": [7.2, 71.3, 7.9, 53.1, 8.1, 7.9, 26.0, 13.0, 7.9, 30.0],
        "Sex": ["male", "female", "female", "female", "male", "male", "female", "male", "female", "male"],
        "Embarked": ["S", "C", "S", "S", None, "Q", "S", "C", "S", "Q"],
    })
    y = pd.Series([0, 1, 1, 1, 0, 0, 1, 0, 1, 0])
    numeric = ["Age", "Fare"]
    categorical = ["Sex", "Embarked"]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
    X_train, X_test, y_train, _ = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    assert len(model.predict(X_test)) == len(X_test)
