import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import PowerTransformer

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}
DATA_NA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20.0, 21.0, 19.0, np.nan],
    "Marks": [0.9, 0.8, 0.7, np.nan],
}

_exp_ls = [0.001, 0.1, 2, 3, 4, 10]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_defo_params_plus_automatically_find_variables(make_df):
    X = make_df(DATA)
    transformer = PowerTransformer(variables=None)
    Xt = transformer.fit_transform(X)

    # test init params
    assert transformer.exp == 0.5
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 4

    # test transform output
    result = nw.from_native(Xt, eager_only=True).to_dict(as_series=False)
    assert result["Age"] == pytest.approx(
        [4.47214, 4.58258, 4.3589, 4.24264], abs=1e-5
    )
    assert result["Marks"] == pytest.approx(
        [0.948683, 0.894427, 0.83666, 0.774597], abs=1e-5
    )

    # inverse transform
    Xit = transformer.inverse_transform(Xt)
    result_it = nw.from_native(Xit, eager_only=True).to_dict(as_series=False)

    # convert numbers to original format.
    assert [round(v) for v in result_it["Age"]] == DATA["Age"]
    assert [round(v, 1) for v in result_it["Marks"]] == DATA["Marks"]


def test_error_if_exp_value_not_allowed():
    with pytest.raises(ValueError):
        PowerTransformer(exp="other")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_na_in_df(make_df):
    X = make_df(DATA_NA)
    with pytest.raises(ValueError):
        transformer = PowerTransformer()
        transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_na_in_df(make_df):
    X = make_df(DATA)
    X_na = make_df(DATA_NA)
    with pytest.raises(ValueError):
        transformer = PowerTransformer()
        transformer.fit(X)
        transformer.transform(X_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    X = make_df(DATA)
    with pytest.raises(NotFittedError):
        transformer = PowerTransformer()
        transformer.transform(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("exp_base", _exp_ls)
def test_inverse_transform_exp_no_default(make_df, exp_base):
    X = make_df(DATA)
    transformer = PowerTransformer(exp=exp_base)
    Xt = transformer.fit_transform(X)
    Xit = transformer.inverse_transform(Xt)

    result_it = nw.from_native(Xit, eager_only=True).to_dict(as_series=False)

    # convert numbers to original format.
    assert [round(v) for v in result_it["Age"]] == DATA["Age"]
    assert [round(v, 1) for v in result_it["Marks"]] == DATA["Marks"]

    # test init params
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 4
