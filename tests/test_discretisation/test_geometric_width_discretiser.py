import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import GeometricWidthDiscretiser


def _normal_dist_data():
    np.random.seed(0)
    mu, sigma = 0, 0.1  # mean and standard deviation
    return {"var": list(np.random.normal(mu, sigma, 100))}


def _get_column_values(X, column):
    return nw.from_native(X, eager_only=True).get_column(column).to_list()


def _get_column_dtype(X, column):
    return nw.from_native(X, eager_only=True).get_column(column).dtype


# test init params
@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_object_not_bool(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(return_object=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_boundaries_not_bool(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(return_boundaries=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 0, -1])
def test_raises_error_when_precision_not_int(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(precision=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}])
def test_raises_error_when_bins_not_int(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(bins=param)


@pytest.mark.parametrize("params", [(False, 1), (True, 10)])
def test_correct_param_assignment_at_init(params):
    param1, param2 = params
    t = GeometricWidthDiscretiser(
        return_object=param1, return_boundaries=param1, precision=param2, bins=param2
    )
    assert t.return_object is param1
    assert t.return_boundaries is param1
    assert t.precision == param2
    assert t.bins == param2


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_and_transform_methods(make_df):
    data = _normal_dist_data()
    df = make_df(data)

    transformer = GeometricWidthDiscretiser(
        bins=10, variables=None, return_object=False
    )
    X = transformer.fit_transform(df)

    # manual calculation
    arr = np.array(data["var"])
    min_, max_ = arr.min(), arr.max()
    increment = np.power(max_ - min_, 1.0 / 10)
    bins = np.r_[-np.inf, min_ + np.power(increment, np.arange(1, 10)), np.inf]
    bins = np.sort(bins)

    # fit params
    assert (transformer.binner_dict_["var"] == bins).all()

    # transform params - ground truth from pandas.cut on the same bins; values
    # must match regardless of which backend the input dataframe uses.
    expected = list(pd.cut(pd.Series(arr), bins=bins, precision=7).cat.codes)
    assert _get_column_values(X, "var") == expected


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables_and_return_as_object(make_df):
    df = make_df(_normal_dist_data())
    transformer = GeometricWidthDiscretiser(bins=10, variables=None, return_object=True)
    X = transformer.fit_transform(df)
    assert _get_column_dtype(X, "var") == nw.Object


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_input_df_contains_na_in_fit(make_df):
    df_na = make_df({"Age": [20.0, 21.0, float("nan"), 23.0]})
    transformer = GeometricWidthDiscretiser()
    with pytest.raises(ValueError):
        transformer.fit(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_input_df_contains_na_in_transform(make_df):
    df = make_df({"Age": [20.0, 21.0, 19.0, 23.0]})
    df_na = make_df({"Age": [20.0, 21.0, float("nan"), 23.0]})

    transformer = GeometricWidthDiscretiser()
    transformer.fit(df)
    with pytest.raises(ValueError):
        transformer.transform(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    df = make_df({"Age": [20.0, 21.0, 19.0, 23.0]})
    transformer = GeometricWidthDiscretiser()
    with pytest.raises(NotFittedError):
        transformer.transform(df)
