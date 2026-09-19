import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import ReciprocalTransformer

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
def test_automatically_find_variables_and_inverse_transform(make_df):
    X = make_df(DATA)
    transformer = ReciprocalTransformer(variables=None)
    Xt = transformer.fit_transform(X)

    # test init params
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 4

    # test transform output
    result = nw.from_native(Xt, eager_only=True).to_dict(as_series=False)
    assert result["Age"] == pytest.approx(
        [0.05, 0.047619, 0.052632, 0.055556], abs=1e-5
    )
    assert result["Marks"] == pytest.approx(
        [1.111111, 1.25, 1.428571, 1.666667], abs=1e-5
    )

    # test inverse_transform
    Xit = transformer.inverse_transform(Xt)
    result_it = nw.from_native(Xit, eager_only=True).to_dict(as_series=False)
    assert [round(v) for v in result_it["Age"]] == DATA["Age"]
    assert [round(v, 1) for v in result_it["Marks"]] == DATA["Marks"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_na_in_df(make_df):
    X = make_df(DATA_NA)
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_na_in_df(make_df):
    X = make_df(DATA)
    X_na = make_df(DATA_NA)
    transformer = ReciprocalTransformer()
    transformer.fit(X)
    with pytest.raises(ValueError):
        transformer.transform(X_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_df_contains_0_as_value(make_df):
    data_zero = dict(DATA)
    data_zero["Age"] = [20, 0, 19, 18]
    X = make_df(DATA)
    X_zero = make_df(data_zero)

    # when variable contains zero, fit
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(X_zero)

    # when variable contains zero, transform
    transformer = ReciprocalTransformer()
    transformer.fit(X)
    with pytest.raises(ValueError):
        transformer.transform(X_zero)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    X = make_df(DATA)
    with pytest.raises(NotFittedError):
        transformer = ReciprocalTransformer()
        transformer.transform(X)
