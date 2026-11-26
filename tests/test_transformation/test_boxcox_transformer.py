import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import BoxCoxTransformer

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}

DATA_NA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, None, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}


def assert_df_equal(X, expected: dict, abs_tol: float = 1e-5) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert result[col] == pytest.approx(values, abs=abs_tol, nan_ok=True)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_finds_variables_and_inverse_transform(make_df):
    df = make_df(DATA)

    transformer = BoxCoxTransformer(variables=None)
    X = transformer.fit_transform(df)

    # test init params
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 4

    expected = dict(DATA)
    expected["Age"] = [9.78731, 10.1666, 9.40189, 9.0099]
    expected["Marks"] = [-0.101687, -0.207092, -0.316843, -0.431788]
    assert_df_equal(X, expected)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)
    result = nw.from_native(Xit, eager_only=True).to_dict(as_series=False)
    assert [round(v) for v in result["Age"]] == DATA["Age"]
    assert [round(v, 1) for v in result["Marks"]] == DATA["Marks"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_df_contains_na(make_df):
    df_na = make_df(DATA_NA)
    transformer = BoxCoxTransformer()
    with pytest.raises(ValueError):
        transformer.fit(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_df_contains_na(make_df):
    df = make_df(DATA)
    df_na = make_df(DATA_NA)
    transformer = BoxCoxTransformer()
    transformer.fit(df)
    with pytest.raises(ValueError):
        transformer.transform(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_df_contains_negative_values(make_df):
    data_neg = {k: list(v) for k, v in DATA.items()}
    data_neg["Age"][1] = -1
    df_neg = make_df(data_neg)
    df = make_df(DATA)

    # when variable contains negative value, fit
    transformer = BoxCoxTransformer()
    with pytest.raises(ValueError):
        transformer.fit(df_neg)

    # when variable contains negative value, transform
    transformer = BoxCoxTransformer()
    transformer.fit(df)
    with pytest.raises(ValueError):
        transformer.transform(df_neg)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    df = make_df(DATA)
    transformer = BoxCoxTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(df)
