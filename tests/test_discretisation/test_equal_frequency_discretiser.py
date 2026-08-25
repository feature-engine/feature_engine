import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import EqualFrequencyDiscretiser


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables_and_return_as_numeric(make_df, df_normal_dist):
    # test case 1: automatically select variables, return_object=False
    data = make_df(df_normal_dist)
    transformer = EqualFrequencyDiscretiser(q=10, variables=None, return_object=False)
    X = transformer.fit_transform(data)

    # output expected for fit attr, computed via pandas.qcut (verified bit-exact
    # against the transformer's own numpy-based bin edges on both backends)
    _, bins = pd.qcut(x=df_normal_dist["var"], q=10, retbins=True, duplicates="drop")
    bins = list(bins)
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")

    # expected transform output
    X_t = [x for x in range(0, 10)]

    # test init params
    assert transformer.q == 10
    assert transformer.variables is None
    assert transformer.return_object is False
    # test fit attr
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    # test transform output
    assert transformer.binner_dict_["var"] == bins
    X_pd = X if isinstance(X, pd.DataFrame) else X.to_pandas()
    assert all(x for x in X_pd["var"].unique() if x not in X_t)
    # in equal frequency discretisation, all intervals get same proportion of values
    assert len((X_pd["var"].value_counts()).unique()) == 1


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables_and_return_as_object(make_df, df_normal_dist):
    # test case 2: return variables cast as object
    data = make_df(df_normal_dist)
    transformer = EqualFrequencyDiscretiser(q=10, variables=None, return_object=True)
    X = transformer.fit_transform(data)
    if isinstance(X, pd.DataFrame):
        assert X["var"].dtypes == "O"
    else:
        assert X["var"].dtype == pl.Object


def test_error_when_q_not_number():
    with pytest.raises(ValueError):
        EqualFrequencyDiscretiser(q="other")


def test_error_if_return_object_not_bool():
    with pytest.raises(ValueError):
        EqualFrequencyDiscretiser(return_object="other")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_input_df_contains_na_in_fit(make_df, df_na):
    # test case 3: when dataset contains na, fit method
    data = make_df(df_na)
    with pytest.raises(ValueError):
        transformer = EqualFrequencyDiscretiser()
        transformer.fit(data)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_input_df_contains_na_in_transform(make_df, df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    fit_data = make_df(df_vartypes)
    transform_data = make_df(df_na[["Name", "City", "Age", "Marks", "dob"]])
    with pytest.raises(ValueError):
        transformer = EqualFrequencyDiscretiser()
        transformer.fit(fit_data)
        transformer.transform(transform_data)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df, df_vartypes):
    data = make_df(df_vartypes)
    with pytest.raises(NotFittedError):
        transformer = EqualFrequencyDiscretiser()
        transformer.transform(data)
