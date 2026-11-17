
import pandas as pd
from src.features import add_features


def test_feature_engineering_adds_family_features():
    sample = pd.DataFrame({
        "Name": ["Smith, Mrs. Jane", "Jones, Mr. John"],
        "SibSp": [1, 0],
        "Parch": [1, 0],
        "Cabin": ["C85", None],
    })
    result = add_features(sample)
    assert result.loc[0, "FamilySize"] == 3
    assert result.loc[1, "IsAlone"] == 1
    assert result.loc[0, "Title"] == "Mrs"
    assert result.loc[0, "CabinDeck"] == "C"
    assert result.loc[1, "CabinDeck"] == "U"


def test_feature_engineering_does_not_mutate_input():
    sample = pd.DataFrame({"SibSp": [0], "Parch": [0]})
    original_columns = list(sample.columns)
    add_features(sample)
    assert list(sample.columns) == original_columns
