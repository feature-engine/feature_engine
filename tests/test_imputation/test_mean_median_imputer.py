import re

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from feature_engine.imputation import MeanImputer, MeanMedianImputer

DEPRECATION_WARNING = (
    "MeanMedianImputer was deprecated in favour of MeanImputer in version "
    "2.0.0 and will be removed in version 2.1.0. To silence this warning, "
    "use MeanImputer instead."
)

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
}


def _cols(X, columns):
    # to_dict(as_series=False) is a convenient, backend-agnostic way to read
    # values back out for comparison, regardless of pandas vs polars.
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    return {c: result[c] for c in columns}


def _null_count(X, col):
    return nw.from_native(X, eager_only=True)[col].null_count()


@pytest.fixture(
    params=[MeanImputer, MeanMedianImputer],
    ids=["MeanImputer", "MeanMedianImputer"],
)
def imputer_class(request):
    return request.param


def make_imputer(imputer_class, **kwargs):
    if imputer_class is MeanMedianImputer:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            return imputer_class(**kwargs)
    return imputer_class(**kwargs)


def test_mean_median_imputer_raises_future_warning():
    with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
        MeanMedianImputer()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_mean_imputation_and_automatically_select_variables(make_df, imputer_class):
    df_na = make_df(DATA)
    imputer = make_imputer(imputer_class, imputation_method="mean", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # test init params
    assert imputer.imputation_method == "mean"
    assert imputer.variables is None

    # test fit attributes
    assert imputer.variables_ == ["Age", "Marks"]
    rounded_dict = {
        key: round(value, 3) for (key, value) in imputer.imputer_dict_.items()
    }
    assert rounded_dict == {"Age": 28.714, "Marks": 0.683}
    assert imputer.n_features_in_ == 5

    # test transform output:
    # selected variables should have no NA
    # not selected variables should still have NA
    assert _null_count(X_transformed, "Age") == 0
    assert _null_count(X_transformed, "Marks") == 0
    assert _null_count(X_transformed, "Name") > 0
    assert _null_count(X_transformed, "City") > 0
    result = _cols(X_transformed, ["Age", "Marks"])
    assert result["Age"] == pytest.approx(
        [20, 21, 19, 28.714285714285715, 23, 40, 41, 37]
    )
    assert result["Marks"] == pytest.approx(
        [0.9, 0.8, 0.7, 0.6833333333333332, 0.3, 0.6833333333333332, 0.8, 0.6]
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_median_imputation_when_user_enters_single_variables(make_df, imputer_class):
    df_na = make_df(DATA)
    imputer = make_imputer(
        imputer_class, imputation_method="median", variables=["Age"]
    )
    X_transformed = imputer.fit_transform(df_na)

    # test init params
    assert imputer.imputation_method == "median"
    assert imputer.variables == ["Age"]

    # test fit attributes
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {"Age": 23.0}

    # test transform output
    assert _null_count(X_transformed, "Age") == 0
    result = _cols(X_transformed, ["Age"])
    assert result["Age"] == [20, 21, 19, 23.0, 23, 40, 41, 37]


def test_error_with_wrong_imputation_method(imputer_class):
    with pytest.raises(ValueError):
        make_imputer(imputer_class, imputation_method="arbitrary")
