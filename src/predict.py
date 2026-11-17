
"""Load the saved end-to-end pipeline and predict survival for passengers."""

from pathlib import Path
import joblib
import pandas as pd

from src.features import add_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "titanic_model.pkl"


def predict_passengers(passengers: pd.DataFrame) -> pd.DataFrame:
    """Return predicted class and survival probability for passengers."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Train the model first: run python -m src.train"
        )

    model = joblib.load(MODEL_PATH)

    # Work with a copy of the input data
    passengers = passengers.copy()

    # Add defaults for optional columns needed by feature engineering
    if "Name" not in passengers.columns:
        passengers["Name"] = "Unknown"

    if "Cabin" not in passengers.columns:
        passengers["Cabin"] = "Unknown"

    # Apply the same feature engineering used during training
    engineered = add_features(passengers)

    # Remove columns not used as model inputs
    drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
        "Survived"
    ]

    X = engineered.drop(
        columns=[c for c in drop if c in engineered.columns]
    )

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "predicted_survived": predictions.astype(int),
        "survival_probability": probabilities
    })


if __name__ == "__main__":

    sample = pd.DataFrame([
        {
            "Pclass": 1,
            "Sex": "female",
            "Age": 29,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 80,
            "Embarked": "S"
        },
        {
            "Pclass": 3,
            "Sex": "male",
            "Age": 30,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 8,
            "Embarked": "S"
        }
    ])

    print("\nPassenger Predictions:")
    print(predict_passengers(sample).to_string(index=False))
