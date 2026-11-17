
"""Feature engineering for the Titanic passenger dataset."""
import pandas as pd


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with engineered features; does not modify the input."""
    df = data.copy()

    if "Name" in df.columns:
        title = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
        title = title.fillna("Unknown").str.strip()
        title = title.replace({
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
            "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare",
            "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare",
            "Dona": "Rare"
        })
        df["Title"] = title

    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    if "Cabin" in df.columns:
        df["CabinDeck"] = df["Cabin"].fillna("Unknown").astype(str).str[0]

    return df
