import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import ArcsinTransformer

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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_and_inverse_transform(make_df):
    X = make_df(DATA)
    transformer = ArcsinTransformer(variables=["Marks"])
    Xt = transformer.fit_transform(X)

    result = nw.from_native(Xt, eager_only=True).to_dict(as_series=False)
    assert result["Marks"] == pytest.approx(
        [1.24905, 1.10715, 0.99116, 0.88607], abs=1e-5
    )

    Xit = transformer.inverse_transform(Xt)
    result_it = nw.from_native(Xit, eager_only=True).to_dict(as_series=False)
    assert [round(v, 1) for v in result_it["Marks"]] == DATA["Marks"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_na_in_df(make_df):
    X = make_df(DATA_NA)
    transformer = ArcsinTransformer(variables=["Marks"])
    with pytest.raises(ValueError):
        transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_na_in_df(make_df):
    X = make_df(DATA)
    X_na = make_df(DATA_NA)
    transformer = ArcsinTransformer(variables=["Marks"])
    transformer.fit(X)
    with pytest.raises(ValueError):
        transformer.transform(X_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_df_contains_outside_range_values(make_df):
    data_out_range = dict(DATA)
    data_out_range["Marks"] = [0.9, 2, 0.7, 0.6]
    X = make_df(DATA)
    X_out_range = make_df(data_out_range)

    transformer = ArcsinTransformer(variables=["Marks"])
    with pytest.raises(ValueError):
        transformer.fit(X_out_range)

    transformer.fit(X)
    with pytest.raises(ValueError):
        transformer.transform(X_out_range)

    transformer = ArcsinTransformer()
    with pytest.raises(ValueError):
        transformer.fit(X_out_range)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    X = make_df(DATA)
    transformer = ArcsinTransformer(variables="Marks")
    with pytest.raises(NotFittedError):
        transformer.transform(X)
