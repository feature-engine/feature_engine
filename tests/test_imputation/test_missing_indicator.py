import re
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.imputation import AddMissingIndicator, MissingIndicator
from tests.backend_helpers import frame_to_dict

INDICATORS = [MissingIndicator, AddMissingIndicator]


# init parameters
@pytest.mark.parametrize("indicator_cls", INDICATORS)
@pytest.mark.parametrize("missing_only", ["missing_only", 1, None])
def test_error_when_missing_only_not_bool(indicator_cls, missing_only):
    msg = f"missing_only takes values True or False. Got {missing_only} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        indicator_cls(missing_only=missing_only)


@pytest.mark.parametrize("indicator_cls", INDICATORS)
@pytest.mark.parametrize("missing_only", [True, False])
def test_init_param_assignment(indicator_cls, missing_only):
    imputer = indicator_cls(missing_only=missing_only)
    assert imputer.missing_only is missing_only


# fit and transform
@pytest.mark.parametrize("indicator_cls", INDICATORS)
def test_detect_variables_with_missing_data_when_variables_is_none(
    make_df, data_na_dob, indicator_cls
):
    # test case 1: automatically detect variables with missing data
    imputer = indicator_cls(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(make_df(data_na_dob))

    # fit params
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 6

    # transform outputs
    assert isinstance(X_transformed, make_df)
    assert X_transformed.shape == (8, 11)
    result = frame_to_dict(X_transformed)
    assert "Name_na" in result
    assert sum(result["Name_na"]) == 2


@pytest.mark.parametrize("indicator_cls", INDICATORS)
def test_add_indicators_to_all_variables_when_variables_is_none(
    make_df, data_na_dob, indicator_cls
):
    imputer = indicator_cls(missing_only=False, variables=None)
    X_transformed = imputer.fit_transform(make_df(data_na_dob))

    assert imputer.variables_ == [
        "Name",
        "City",
        "Studies",
        "Age",
        "Marks",
        "dob",
    ]
    assert isinstance(X_transformed, make_df)
    assert X_transformed.shape == (8, 12)
    result = frame_to_dict(X_transformed)
    assert "dob_na" in result
    assert sum(result["dob_na"]) == 0


@pytest.mark.parametrize("indicator_cls", INDICATORS)
def test_add_indicators_to_one_variable(make_df, data_na_dob, indicator_cls):
    imputer = indicator_cls(variables="Name")
    X_transformed = imputer.fit_transform(make_df(data_na_dob))

    assert imputer.variables_ == ["Name"]
    assert isinstance(X_transformed, make_df)
    assert X_transformed.shape == (8, 7)
    result = frame_to_dict(X_transformed)
    assert "Name_na" in result
    assert sum(result["Name_na"]) == 2


@pytest.mark.parametrize("indicator_cls", INDICATORS)
def test_detect_variables_with_missing_data_in_variables_entered_by_user(
    make_df, data_na_dob, indicator_cls
):
    imputer = indicator_cls(
        missing_only=True,
        variables=["City", "Studies", "Age", "dob"],
    )
    X_transformed = imputer.fit_transform(make_df(data_na_dob))

    assert imputer.variables_ == ["City", "Studies", "Age"]
    assert isinstance(X_transformed, make_df)
    assert X_transformed.shape == (8, 9)
    result = frame_to_dict(X_transformed)
    assert "City_na" in result
    assert "dob_na" not in result
    assert sum(result["City_na"]) == 2


@pytest.mark.parametrize("indicator_cls", INDICATORS)
def test_get_feature_names_out(make_df, data_na_dob, indicator_cls):
    X = make_df(data_na_dob)
    original_features = list(data_na_dob)

    tr = indicator_cls(missing_only=False)
    tr.fit(X)

    out = [f + "_na" for f in original_features]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out

    tr = indicator_cls(missing_only=True)
    tr.fit(X)

    out = [f + "_na" for f in original_features[0:-1]]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out

    with pytest.raises(ValueError):
        tr.get_feature_names_out("Name")

    with pytest.raises(ValueError):
        tr.get_feature_names_out(["Name", "hola"])


@pytest.mark.parametrize("indicator_cls", INDICATORS)
def test_get_feature_names_out_from_pipeline(make_df, data_na_dob, indicator_cls):
    X = make_df(data_na_dob)
    original_features = list(data_na_dob)

    tr = Pipeline([("transformer", indicator_cls(missing_only=False))])
    tr.fit(X)

    out = [f + "_na" for f in original_features]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out


@pytest.mark.parametrize("indicator_cls", INDICATORS)
def test_no_performance_warning_with_many_variables(indicator_cls):
    # pandas-only.
    n_cols = 101

    df = pd.DataFrame(
        np.random.randn(10, n_cols),
        columns=[f"col_{i}" for i in range(n_cols)],
    )

    # Introduce missing values
    df.iloc[0, :] = np.nan

    ami = indicator_cls(missing_only=False)
    ami.fit(df)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ami.transform(df)

    assert not any(
        issubclass(w.category, pd.errors.PerformanceWarning)
        for w in captured
    ), "PerformanceWarning was raised during transform"


def test_add_missing_indicator_deprecation_warning():
    with pytest.warns(
        FutureWarning,
        match="Use MissingIndicator instead",
    ):
        AddMissingIndicator()
