import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import YeoJohnsonTransformer

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
def test_automatically_select_variables_and_inverse_transform(make_df):
    X = make_df(DATA)
    transformer = YeoJohnsonTransformer(variables=None)
    Xt = transformer.fit_transform(X)

    # test init params
    assert transformer.variables is None
    # test fit attrs
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 4

    # test transform output
    result = nw.from_native(Xt, eager_only=True).to_dict(as_series=False)
    assert result["Age"] == pytest.approx(
        [10.167048, 10.540602, 9.787738, 9.402289], abs=1e-5
    )
    assert result["Marks"] == pytest.approx(
        [0.804449, 0.722367, 0.638807, 0.553652], abs=1e-5
    )

    # test inverse_transform, including non-transformed columns
    Xit = transformer.inverse_transform(Xt)
    result_it = nw.from_native(Xit, eager_only=True).to_dict(as_series=False)
    assert [round(v) for v in result_it["Age"]] == DATA["Age"]
    assert [round(v, 1) for v in result_it["Marks"]] == DATA["Marks"]
    assert result_it["Name"] == DATA["Name"]
    assert result_it["City"] == DATA["City"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transformer_on_integer_variables(make_df):
    X = make_df(
        {
            "var1": [0, 1, 0, 2, 3, 4, 5, 6, 8, 10],
            "var2": [12, 11, 10, 15, 13, 12, 11, 10, 10, 20],
        }
    )

    Xt = YeoJohnsonTransformer().fit_transform(X)
    result = nw.from_native(Xt, eager_only=True).to_dict(as_series=False)

    assert result["var1"] == pytest.approx(
        [
            0.0,
            0.787147,
            0.0,
            1.347166,
            1.797028,
            2.179455,
            2.515513,
            2.817345,
            3.346739,
            3.805171,
        ],
        abs=1e-5,
    )
    assert result["var2"] == pytest.approx(
        [
            0.289101,
            0.289088,
            0.289069,
            0.289121,
            0.289110,
            0.289101,
            0.289088,
            0.289069,
            0.289069,
            0.289133,
        ],
        abs=1e-5,
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_na_in_df(make_df):
    X = make_df(DATA_NA)
    with pytest.raises(ValueError):
        transformer = YeoJohnsonTransformer()
        transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_na_in_df(make_df):
    X = make_df(DATA)
    X_na = make_df(DATA_NA)
    transformer = YeoJohnsonTransformer()
    transformer.fit(X)
    with pytest.raises(ValueError):
        transformer.transform(X_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    X = make_df(DATA)
    with pytest.raises(NotFittedError):
        transformer = YeoJohnsonTransformer()
        transformer.transform(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_with_x_negative_and_positive(make_df):
    X = make_df(
        {
            "var1": list(np.arange(-20, 0)),
            "var2": list(np.arange(0, 20)),
            "var3": list(np.arange(-10, 10)),
        }
    )

    transformer = YeoJohnsonTransformer(variables=None)
    Xt = transformer.fit_transform(X)
    Xi = transformer.inverse_transform(Xt)
    result = nw.from_native(Xi, eager_only=True).to_dict(as_series=False)

    assert [round(v) for v in result["var1"]] == list(np.arange(-20, 0))
    assert [round(v) for v in result["var2"]] == list(np.arange(0, 20))
    assert [round(v) for v in result["var3"]] == list(np.arange(-10, 10))


def test_inverse_with_non_linear_index():
    # pandas-specific: exercises index-preserving behaviour, which has no
    # polars equivalent (polars has no row index).
    X = pd.DataFrame(
        {
            "var1": np.arange(-20, 0),
            "var2": np.arange(0, 20),
            "var3": np.arange(-10, 10),
        },
        index=[13, 15, 12, 11, 17, 9, 4, 0, 1, 14, 18, 2, 3, 6, 5, 7, 8, 2, 16, 10],
    )

    transformer = YeoJohnsonTransformer(variables=None)
    Xt = transformer.fit_transform(X)

    Xi = transformer.inverse_transform(Xt)
    Xi = Xi.round(0).astype(int)

    pd.testing.assert_frame_equal(X, Xi, check_dtype=False)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_lambda_equal_0(make_df):
    X = make_df({"var1": list(np.arange(0, 20)), "var2": list(np.arange(20, 40))})

    transformer = YeoJohnsonTransformer(variables=None)
    transformer = transformer.fit(X)
    transformer.lambda_dict_ = {"var1": 0, "var2": 0}

    Xt = transformer.transform(X)
    Xi = transformer.inverse_transform(Xt)
    result = nw.from_native(Xi, eager_only=True).to_dict(as_series=False)

    assert [round(v) for v in result["var1"]] == list(np.arange(0, 20))
    assert [round(v) for v in result["var2"]] == list(np.arange(20, 40))


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_lambda_equal_2(make_df):
    X = make_df({"var1": list(np.arange(-21, -1)), "var2": list(np.arange(-41, -21))})

    transformer = YeoJohnsonTransformer(variables=None)
    transformer = transformer.fit(X)
    transformer.lambda_dict_ = {"var1": 2, "var2": 2}

    Xt = transformer.transform(X)
    Xi = transformer.inverse_transform(Xt)
    result = nw.from_native(Xi, eager_only=True).to_dict(as_series=False)

    assert [round(v) for v in result["var1"]] == list(np.arange(-21, -1))
    assert [round(v) for v in result["var2"]] == list(np.arange(-41, -21))
