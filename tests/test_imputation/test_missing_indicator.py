import warnings
import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from feature_engine.imputation import AddMissingIndicator


def test_detect_variables_with_missing_data_when_variables_is_none(df_na):
    # test case 1: automatically detect variables with missing data
    imputer = AddMissingIndicator(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # init params
    assert imputer.missing_only is True
    assert imputer.variables is None
    # fit params
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 6
    # transform outputs
    assert X_transformed.shape == (8, 11)
    assert "Name_na" in X_transformed.columns
    assert X_transformed["Name_na"].sum() == 2


def test_add_indicators_to_all_variables_when_variables_is_none(df_na):
    imputer = AddMissingIndicator(missing_only=False, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks", "dob"]
    assert X_transformed.shape == (8, 12)
    assert "dob_na" in X_transformed.columns
    assert X_transformed["dob_na"].sum() == 0


def test_add_indicators_to_one_variable(df_na):
    imputer = AddMissingIndicator(missing_only=False, variables="Name")
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables_ == ["Name"]
    assert X_transformed.shape == (8, 7)
    assert "Name_na" in X_transformed.columns
    assert X_transformed["Name_na"].sum() == 2


def test_detect_variables_with_missing_data_in_variables_entered_by_user(df_na):
    imputer = AddMissingIndicator(
        missing_only=False, variables=["City", "Studies", "Age", "dob"]
    )
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables == ["City", "Studies", "Age", "dob"]
    assert imputer.variables_ == ["City", "Studies", "Age", "dob"]
    assert X_transformed.shape == (8, 10)
    assert "City_na" in X_transformed.columns
    assert "dob_na" in X_transformed.columns
    assert X_transformed["City_na"].sum() == 2


def test_error_when_missing_only_not_bool():
    with pytest.raises(ValueError):
        AddMissingIndicator(missing_only="missing_only")


def test_get_feature_names_out(df_na):
    original_features = df_na.columns.to_list()

    tr = AddMissingIndicator(missing_only=False)
    tr.fit(df_na)

    out = [f + "_na" for f in original_features]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out

    tr = AddMissingIndicator(missing_only=True)
    tr.fit(df_na)

    out = [f + "_na" for f in original_features[0:-1]]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out

    with pytest.raises(ValueError):
        tr.get_feature_names_out("Name")

    with pytest.raises(ValueError):
        tr.get_feature_names_out(["Name", "hola"])


def test_get_feature_names_out_from_pipeline(df_na):
    original_features = df_na.columns.to_list()

    tr = Pipeline([("transformer", AddMissingIndicator(missing_only=False))])
    tr.fit(df_na)

    out = [f + "_na" for f in original_features]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out


def test_no_performance_warning_with_many_variables():
    n_cols = 101
    df = pd.DataFrame(
        np.random.randn(10, n_cols),
        columns=[f"col_{i}" for i in range(n_cols)],
    )

    # Introduce missing values
    df.iloc[0, :] = np.nan

    ami = AddMissingIndicator(missing_only=False)
    ami.fit(df)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ami.transform(df)

    assert not any(
        issubclass(w.category, pd.errors.PerformanceWarning)
        for w in captured
    ), "PerformanceWarning was raised during transform"


# ---------------------------------------------------------------------------
# Tests for variables + missing_only mutual exclusivity
# ---------------------------------------------------------------------------

def test_error_when_variables_and_missing_only_true():
    """Passing both variables and missing_only=True should raise ValueError."""
    with pytest.raises(ValueError, match="variables and missing_only"):
        AddMissingIndicator(missing_only=True, variables=["Age", "Name"])


def test_no_error_when_variables_and_missing_only_false():
    """variables + missing_only=False is valid — should not raise."""
    imputer = AddMissingIndicator(missing_only=False, variables=["Age"])
    assert imputer.variables is not None


def test_no_error_when_variables_none_and_missing_only_true():
    """variables=None + missing_only=True is valid — default case."""
    imputer = AddMissingIndicator(missing_only=True, variables=None)
    assert imputer.missing_only is True
