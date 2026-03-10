import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.scaling import GroupStandardScaler

def test_group_standard_scaler_single_reference():
    # Input data
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "var2": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
            "grp": ["A", "A", "A", "B", "B", "B"],
        }
    )
    
    # Expected Output
    # Group A: var1 mean=2, std=1; var2 mean=5, std=1
    # Group B: var1 mean=20, std=10; var2 mean=50, std=10
    expected_df = pd.DataFrame(
        {
            "var1": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            "var2": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            "grp": ["A", "A", "A", "B", "B", "B"],
        }
    )

    transformer = GroupStandardScaler(variables=["var1", "var2"], reference=["grp"])
    X = transformer.fit_transform(df)

    pd.testing.assert_frame_equal(X, expected_df)

    # Test attributes
    assert transformer.variables_ == ["var1", "var2"]
    assert transformer.reference_ == ["grp"]
    assert transformer.barycenter_ == {
        "var1": {"A": 2.0, "B": 20.0},
        "var2": {"A": 5.0, "B": 50.0},
    }
    assert transformer.scale_ == {
        "var1": {"A": 1.0, "B": 10.0},
        "var2": {"A": 1.0, "B": 10.0},
    }

def test_unseen_groups():
    df_train = pd.DataFrame({
        "var1": [2.0, 4.0, 10.0, 20.0],
        "grp": ["A", "A", "B", "B"]
    })
    
    # Group A var1: mean=3, std=1.414
    # Group B var1: mean=15, std=7.07
    # Global var1: mean=9, std=8.165
    
    transformer = GroupStandardScaler(variables=["var1"], reference=["grp"])
    transformer.fit(df_train)

    df_test = pd.DataFrame({
        "var1": [3.0, 15.0, 9.0],
        "grp": ["A", "B", "C"]  # C is unseen
    })
    
    X = transformer.transform(df_test)
    
    # Expected calculation
    # A (seen directly) : (3 - 3) / 1.414 = 0
    # B (seen directly) : (15 - 15) / 7.07 = 0
    # C (unseen, falls back to global): (9 - 9) / 8.165 = 0
    
    expected_df = pd.DataFrame({
        "var1": [0.0, 0.0, 0.0],
        "grp": ["A", "B", "C"]
    })

    pd.testing.assert_frame_equal(X, expected_df)

def test_overlapping_variable_and_reference():
    df = pd.DataFrame({"var1": [1.0, 2.0], "grp": ["A", "B"]})
    transformer = GroupStandardScaler(variables=["var1"], reference=["var1"])
    with pytest.raises(ValueError):
        transformer.fit(df)

def test_non_fitted_error():
    df = pd.DataFrame({"var1": [1.0, 2.0], "grp": ["A", "B"]})
    transformer = GroupStandardScaler(reference=["grp"])
    with pytest.raises(NotFittedError):
        transformer.transform(df)

def test_missing_reference_param():
    with pytest.raises(ValueError, match="Parameter `reference` must be provided."):
        GroupStandardScaler(variables=["var1"])

def test_dataset_contains_na():
    df_na = pd.DataFrame({
        "var1": [1.0, float('nan'), 3.0],
        "grp": ["A", "A", "B"]
    })
    transformer = GroupStandardScaler(reference=["grp"])
    with pytest.raises(ValueError):
        transformer.fit(df_na)
