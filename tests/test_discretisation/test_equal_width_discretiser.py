import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import EqualWidthDiscretiser


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables_and_return_as_numeric(df_normal_dist, make_df):
    transformer = EqualWidthDiscretiser(bins=10, variables=None, return_object=False)
    X = transformer.fit_transform(make_df(df_normal_dist))

    # ground truth bin edges via pandas.cut, same widening/duplicates-drop
    # rules the new fit() replicates in plain numpy.
    _, bins = pd.cut(x=df_normal_dist["var"], bins=10, retbins=True, duplicates="drop")
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")

    expected_codes = pd.cut(
        df_normal_dist["var"], bins=list(bins), labels=False, include_lowest=True
    ).to_numpy()

    # init params
    assert transformer.bins == 10
    assert transformer.variables is None
    assert transformer.return_object is False
    # fit params
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    assert np.allclose(transformer.binner_dict_["var"], bins)
    # transform params: same bin codes on both backends
    assert np.array_equal(np.asarray(X["var"]), expected_codes)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables_and_return_as_object(df_normal_dist, make_df):
    transformer = EqualWidthDiscretiser(bins=10, variables=None, return_object=True)
    X = transformer.fit_transform(make_df(df_normal_dist))

    if isinstance(X, pd.DataFrame):
        assert X["var"].dtype == object
    else:
        assert X["var"].dtype == pl.Object


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_constant_variable_produces_single_bin(make_df):
    # fit()'s bin-edge widening for a zero-range variable (mn == mx): should
    # still fit without error and place every value in the same bin, same as
    # pandas.cut(bins=10) on a constant series.
    df = pd.DataFrame({"var": [5.0] * 10})
    transformer = EqualWidthDiscretiser(bins=10)
    X = transformer.fit_transform(make_df(df))

    _, bins = pd.cut(x=df["var"], bins=10, retbins=True, duplicates="drop")
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")
    expected_codes = pd.cut(
        df["var"], bins=list(bins), labels=False, include_lowest=True
    ).to_numpy()

    assert transformer.binner_dict_["var"][0] == float("-inf")
    assert transformer.binner_dict_["var"][-1] == float("inf")
    assert np.array_equal(np.asarray(X["var"]), expected_codes)


def test_error_when_bins_not_number():
    with pytest.raises(ValueError, match="bins must be an integer"):
        EqualWidthDiscretiser(bins="other")


def test_error_if_return_object_not_bool():
    with pytest.raises(ValueError, match="return_object must be True or False"):
        EqualWidthDiscretiser(return_object="other")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_input_df_contains_na_in_fit(df_na, make_df):
    with pytest.raises(ValueError):
        transformer = EqualWidthDiscretiser()
        transformer.fit(make_df(df_na))


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na, make_df):
    with pytest.raises(ValueError):
        transformer = EqualWidthDiscretiser()
        transformer.fit(make_df(df_vartypes))
        transformer.transform(make_df(df_na[["Name", "City", "Age", "Marks", "dob"]]))


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(df_vartypes, make_df):
    with pytest.raises(NotFittedError):
        transformer = EqualWidthDiscretiser()
        transformer.transform(make_df(df_vartypes))
