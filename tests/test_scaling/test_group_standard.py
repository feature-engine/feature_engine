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


def test_variables_none_auto_detect():
    """Test fit with variables=None auto-detects numerical and excludes reference."""
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "var2": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
            "grp": ["A", "A", "A", "B", "B", "B"],
        }
    )
    transformer = GroupStandardScaler(reference=["grp"])
    transformer.fit(df)
    assert transformer.variables_ == ["var1", "var2"]
    X = transformer.transform(df)
    assert list(X.columns) == transformer.get_feature_names_out()


def test_single_element_group_zero_std():
    """Test groups with one element or zero std (scale 0, replaced in transform)."""
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 2.0, 2.0],
            "grp": ["A", "B", "B", "B"],
        }
    )
    # Group A: single element -> std NaN -> set to 0 in fit
    # Group B: constant 2.0 -> std 0
    transformer = GroupStandardScaler(variables=["var1"], reference=["grp"])
    transformer.fit(df)
    assert transformer.scale_["var1"]["A"] == 0.0
    assert transformer.scale_["var1"]["B"] == 0.0
    X = transformer.transform(df)
    # In transform, 0 std is replaced with 1, so (x - mean) / 1
    assert "var1" in X.columns


def test_global_std_zero_or_nan():
    """Test global std NaN/0 fallback to 1.0 for unseen groups."""
    df = pd.DataFrame({"var1": [5.0, 5.0, 5.0], "grp": ["A", "A", "B"]})
    transformer = GroupStandardScaler(variables=["var1"], reference=["grp"])
    transformer.fit(df)
    assert transformer.global_std_["var1"] == 1.0
    df_test = pd.DataFrame({"var1": [5.0], "grp": ["C"]})
    X = transformer.transform(df_test)
    pd.testing.assert_frame_equal(
        X, pd.DataFrame({"var1": [0.0], "grp": ["C"]})
    )


def test_multiple_reference_columns():
    """Test transform with multiple reference variables (group_keys from zip)."""
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "r1": ["X", "X", "X", "Y", "Y", "Y"],
            "r2": ["a", "a", "b", "a", "b", "b"],
        }
    )
    transformer = GroupStandardScaler(
        variables=["var1"], reference=["r1", "r2"]
    )
    transformer.fit(df)
    X = transformer.transform(df)
    assert X.shape[0] == 6
    assert list(X.columns) == transformer.get_feature_names_out()


def test_get_feature_names_out():
    """Test get_feature_names_out returns feature names in order."""
    df = pd.DataFrame(
        {"var1": [1.0, 2.0], "var2": [3.0, 4.0], "grp": ["A", "B"]}
    )
    transformer = GroupStandardScaler(
        variables=["var1", "var2"], reference=["grp"]
    )
    transformer.fit(df)
    names = transformer.get_feature_names_out()
    assert names == ["var1", "var2", "grp"]


def test_more_tags():
    """Test transformer tags for sklearn compatibility."""
    gss = GroupStandardScaler(variables=["x"], reference=["g"])
    tags = gss._more_tags()
    assert tags["variables"] == "numerical"
    assert "check_parameters_default_constructible" in tags["_xfail_checks"]
