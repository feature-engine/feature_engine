import datetime
import warnings

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklearn.pipeline import Pipeline

from feature_engine.imputation import MissingIndicator, AddMissingIndicator

DATA = {
    "Name": ["tom", "nick", "krish", None, "peter", None, "fred", "sam"],
    "City": [
        "London",
        "Manchester",
        None,
        None,
        "London",
        "London",
        "Bristol",
        "Manchester",
    ],
    "Studies": [
        "Bachelor",
        "Bachelor",
        None,
        None,
        "Bachelor",
        "PhD",
        "None",
        "Masters",
    ],
    "Age": [20, 21, 19, None, 23, 40, 41, 37],
    "Marks": [0.9, 0.8, 0.7, None, 0.3, None, 0.8, 0.6],
    "dob": [
        datetime.datetime(2020, 2, 24) + datetime.timedelta(minutes=i)
        for i in range(8)
    ],
}


def _cols(X):
    return list(nw.from_native(X, eager_only=True).columns)


def _col_sum(X, col):
    return sum(nw.from_native(X, eager_only=True).get_column(col).to_list())


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "indicator_cls",
    [MissingIndicator, AddMissingIndicator],
)
def test_detect_variables_with_missing_data_when_variables_is_none(
    make_df, indicator_cls
):
    X = make_df(DATA)
    # test case 1: automatically detect variables with missing data
    imputer = indicator_cls(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(X)

    # init params
    assert imputer.missing_only is True
    assert imputer.variables is None

    # fit params
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 6

    # transform outputs
    assert X_transformed.shape == (8, 11)
    assert "Name_na" in _cols(X_transformed)
    assert _col_sum(X_transformed, "Name_na") == 2


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "indicator_cls",
    [MissingIndicator, AddMissingIndicator],
)
def test_add_indicators_to_all_variables_when_variables_is_none(
    make_df, indicator_cls
):
    X = make_df(DATA)
    imputer = indicator_cls(missing_only=False, variables=None)

    X_transformed = imputer.fit_transform(X)

    assert imputer.variables_ == [
        "Name",
        "City",
        "Studies",
        "Age",
        "Marks",
        "dob",
    ]
    assert X_transformed.shape == (8, 12)
    assert "dob_na" in _cols(X_transformed)
    assert _col_sum(X_transformed, "dob_na") == 0


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "indicator_cls",
    [MissingIndicator, AddMissingIndicator],
)
def test_add_indicators_to_one_variable(make_df, indicator_cls):
    X = make_df(DATA)
    imputer = indicator_cls(variables="Name")

    X_transformed = imputer.fit_transform(X)

    assert imputer.variables_ == ["Name"]
    assert X_transformed.shape == (8, 7)
    assert "Name_na" in _cols(X_transformed)
    assert _col_sum(X_transformed, "Name_na") == 2


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "indicator_cls",
    [MissingIndicator, AddMissingIndicator],
)
def test_detect_variables_with_missing_data_in_variables_entered_by_user(
    make_df, indicator_cls
):
    X = make_df(DATA)
    imputer = indicator_cls(
        missing_only=True,
        variables=["City", "Studies", "Age", "dob"],
    )

    X_transformed = imputer.fit_transform(X)

    assert imputer.variables == ["City", "Studies", "Age", "dob"]
    assert imputer.variables_ == ["City", "Studies", "Age"]
    assert X_transformed.shape == (8, 9)
    assert "City_na" in _cols(X_transformed)
    assert "dob_na" not in _cols(X_transformed)
    assert _col_sum(X_transformed, "City_na") == 2


@pytest.mark.parametrize(
    "indicator_cls",
    [MissingIndicator, AddMissingIndicator],
)
def test_error_when_missing_only_not_bool(indicator_cls):
    with pytest.raises(ValueError):
        indicator_cls(missing_only="missing_only")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "indicator_cls",
    [MissingIndicator, AddMissingIndicator],
)
def test_get_feature_names_out(make_df, indicator_cls):
    X = make_df(DATA)
    original_features = _cols(X)

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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "indicator_cls",
    [MissingIndicator, AddMissingIndicator],
)
def test_get_feature_names_out_from_pipeline(make_df, indicator_cls):
    X = make_df(DATA)
    original_features = _cols(X)

    tr = Pipeline(
        [("transformer", indicator_cls(missing_only=False))]
    )

    tr.fit(X)

    out = [f + "_na" for f in original_features]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out


@pytest.mark.parametrize(
    "indicator_cls",
    [MissingIndicator, AddMissingIndicator],
)
def test_no_performance_warning_with_many_variables(indicator_cls):
    # pandas-only: exercises the pandas fast path's PerformanceWarning
    # behaviour specifically, not a cross-backend value comparison.
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
