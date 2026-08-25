import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.scaling import MeanNormalisationScaler, MeanNormalizationScaler
from tests.estimator_checks.fit_functionality_checks import check_return_empty
from tests.estimator_checks.non_fitted_error_checks import (
    check_raises_non_fitted_error_when_fit_fails,
)

DEPRECATION_WARNING = (
    "MeanNormalizationScaler was deprecated in favour of "
    "MeanNormalisationScaler in version 2.0.0 and will be removed in version 2.1.0. "
    "To silence this warning, use MeanNormalisationScaler instead."
)

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}


def _none_to_nan(values):
    # Missing values print as None for polars, NaN for pandas float columns
    # - both mean "missing" here, so normalize both sides before comparing.
    return [np.nan if v is None else v for v in values]


def assert_df_equal(X, expected: dict, abs_tol: float = 1e-4) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert _none_to_nan(result[col]) == pytest.approx(
            _none_to_nan(values), abs=abs_tol, nan_ok=True
        )


@pytest.fixture(
    params=[MeanNormalisationScaler, MeanNormalizationScaler],
    ids=["MeanNormalisationScaler", "MeanNormalizationScaler"],
)
def transformer_class(request):
    return request.param


def make_transformer(transformer_class, **kwargs):
    if transformer_class is MeanNormalizationScaler:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            return transformer_class(**kwargs)
    return transformer_class(**kwargs)


def test_mean_normalization_scaler_raises_future_warning():
    with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
        MeanNormalizationScaler()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transforming_int_vars(make_df, transformer_class):
    df = make_df(
        {
            "var1": [1.0, 2.0, 3.0],
            "var2": [4.0, 5.0, 3.0],
            "var3": [40.0, 20.0, 30.0],
        }
    )
    expected = {
        "var1": [-0.5, 0.0, 0.5],
        "var2": [0, 0.5, -0.5],
        "var3": [0.5, -0.5, 0.0],
    }

    transformer = make_transformer(transformer_class, variables=None)
    X = transformer.fit_transform(df)
    assert_df_equal(X, expected)

    Xit = transformer.inverse_transform(X)
    assert_df_equal(
        Xit,
        {"var1": [1.0, 2.0, 3.0], "var2": [4.0, 5.0, 3.0], "var3": [40.0, 20.0, 30.0]},
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_mean_normalization_plus_automatically_find_variables(
    make_df, transformer_class
):
    df = make_df(DATA)

    transformer = make_transformer(transformer_class, variables=None)
    X = transformer.fit_transform(df)

    assert transformer.variables is None
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 4

    expected = dict(DATA)
    expected["Age"] = [0.16667, 0.5, -0.16667, -0.5]
    expected["Marks"] = [0.5, 0.16667, -0.16667, -0.5]
    assert_df_equal(X, expected)

    Xit = transformer.inverse_transform(X)
    assert_df_equal(Xit, DATA)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_mean_normalization_plus_user_passes_var_list(make_df, transformer_class):
    df = make_df(DATA)

    transformer = make_transformer(transformer_class, variables="Age")
    X = transformer.fit_transform(df)

    assert transformer.variables == "Age"
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 4

    expected = dict(DATA)
    expected["Age"] = [0.16667, 0.5, -0.16667, -0.5]
    assert_df_equal(X, expected)

    Xit = transformer.inverse_transform(X)
    assert_df_equal(Xit, DATA)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_na_in_df(make_df, transformer_class):
    data_na = dict(DATA)
    data_na["Age"] = [20, None, 19, 18]
    df_na = make_df(data_na)

    transformer = make_transformer(transformer_class)
    with pytest.raises(ValueError):
        transformer.fit(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_na_in_df(make_df, transformer_class):
    data_na = dict(DATA)
    data_na["Age"] = [20, None, 19, 18]
    df_na = make_df(data_na)

    transformer = make_transformer(transformer_class)
    transformer.fit(make_df(DATA))
    with pytest.raises(ValueError):
        transformer.transform(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df, transformer_class):
    df = make_df(DATA)
    transformer = make_transformer(transformer_class)
    with pytest.raises(NotFittedError):
        transformer.transform(df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_constant_columns_error(make_df, transformer_class):
    df = make_df(
        {
            "var1": [1.0, 2.0, 3.0],
            "var2": [4.0, 5.0, 3.0],
            "var3": [7.0, 7.0, 7.0],
        }
    )

    transformer = make_transformer(transformer_class)
    with pytest.raises(ValueError, match=re.escape("Division by zero is not allowed")):
        transformer.fit(df)


def test_raises_non_fitted_error_when_error_during_fit(transformer_class):
    # constant column: fails after mean_/range_ would have been computed, at
    # the "check for constant columns" step - real regression guard for the
    # deferred trailing-underscore attribute assignment. Pandas-only: this
    # check's own helper (check_raises_non_fitted_error_when_fit_fails)
    # builds a pandas frame internally.
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0],
            "var2": [4.0, 5.0, 3.0],
            "var3": [7.0, 7.0, 7.0],
        }
    )
    transformer = make_transformer(transformer_class)
    check_raises_non_fitted_error_when_fit_fails(transformer, df)


def test_check_return_empty(transformer_class):
    # check_return_empty itself is pandas-only (builds pd.DataFrame internally).
    transformer = make_transformer(transformer_class)
    if transformer_class is MeanNormalizationScaler:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            check_return_empty(transformer)
    else:
        check_return_empty(transformer)
