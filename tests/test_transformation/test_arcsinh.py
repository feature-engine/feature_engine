import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.transformation import ArcSinhTransformer

DATA_NUMERICAL = {
    "a": [-100.0, -10.0, 0.0, 10.0, 100.0],
    "b": [1.0, 2.0, 3.0, 4.0, 5.0],
}
DATA_MULTI_COLUMN = {
    "a": [1.0, 2.0, 3.0],
    "b": [4.0, 5.0, 6.0],
    "c": [7.0, 8.0, 9.0],
}


def _col(X, name):
    return nw.from_native(X, eager_only=True).get_column(name).to_numpy()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_default_parameters(make_df):
    """Test transformer with default parameters applies arcsinh to all columns."""
    X = make_df(DATA_NUMERICAL)
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X)

    expected_a = np.arcsinh(np.array(DATA_NUMERICAL["a"]))
    expected_b = np.arcsinh(np.array(DATA_NUMERICAL["b"]))
    np.testing.assert_array_almost_equal(_col(X_tr, "a"), expected_a)
    np.testing.assert_array_almost_equal(_col(X_tr, "b"), expected_b)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_specific_variables(make_df):
    """Test transformer with specific variables selected."""
    X = make_df(DATA_MULTI_COLUMN)
    transformer = ArcSinhTransformer(variables=["a", "b"])
    X_tr = transformer.fit_transform(X)

    np.testing.assert_array_almost_equal(
        _col(X_tr, "a"), np.arcsinh(np.array(DATA_MULTI_COLUMN["a"]))
    )
    np.testing.assert_array_almost_equal(
        _col(X_tr, "b"), np.arcsinh(np.array(DATA_MULTI_COLUMN["b"]))
    )
    np.testing.assert_array_equal(_col(X_tr, "c"), np.array(DATA_MULTI_COLUMN["c"]))


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_with_loc_and_scale(make_df):
    """Test transformer with loc and scale parameters."""
    data = {"a": [10.0, 20.0, 30.0, 40.0, 50.0]}
    X = make_df(data)
    loc = 30.0
    scale = 10.0
    transformer = ArcSinhTransformer(loc=loc, scale=scale)
    X_tr = transformer.fit_transform(X)

    expected = np.arcsinh((np.array(data["a"]) - loc) / scale)
    np.testing.assert_array_almost_equal(_col(X_tr, "a"), expected)
    np.testing.assert_almost_equal(_col(X_tr, "a")[2], 0.0, decimal=10)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("loc", [0.0, 10.0, -10.0, 100.5])
def test_various_loc_values(make_df, loc):
    """Test that various loc values work correctly."""
    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0]}
    X = make_df(data)
    transformer = ArcSinhTransformer(loc=loc)
    X_tr = transformer.fit_transform(X)

    expected = np.arcsinh((np.array(data["a"]) - loc) / 1.0)
    np.testing.assert_array_almost_equal(_col(X_tr, "a"), expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 10.0, 100.0])
def test_various_scale_values(make_df, scale):
    """Test that various scale values work correctly."""
    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0]}
    X = make_df(data)
    transformer = ArcSinhTransformer(scale=scale)
    X_tr = transformer.fit_transform(X)

    expected = np.arcsinh((np.array(data["a"]) - 0.0) / scale)
    np.testing.assert_array_almost_equal(_col(X_tr, "a"), expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform(make_df):
    """Test inverse_transform returns original values."""
    X = make_df(DATA_NUMERICAL)
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X)
    X_inv = transformer.inverse_transform(X_tr)

    np.testing.assert_array_almost_equal(
        _col(X_inv, "a"), np.array(DATA_NUMERICAL["a"]), decimal=10
    )
    np.testing.assert_array_almost_equal(
        _col(X_inv, "b"), np.array(DATA_NUMERICAL["b"]), decimal=10
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_with_loc_scale(make_df):
    """Test inverse_transform with loc and scale parameters."""
    data = {"a": [10.0, 20.0, 30.0, 40.0, 50.0]}
    X = make_df(data)
    transformer = ArcSinhTransformer(loc=25.0, scale=5.0)
    X_tr = transformer.fit_transform(X)
    X_inv = transformer.inverse_transform(X_tr)

    np.testing.assert_array_almost_equal(
        _col(X_inv, "a"), np.array(data["a"]), decimal=10
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_negative_values(make_df):
    """Test that transformer handles negative values correctly."""
    data = {"a": [-1000.0, -500.0, 0.0, 500.0, 1000.0]}
    X = make_df(data)
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X)

    # Expected values: arcsinh([ -1000, -500, 0, 500, 1000 ])
    expected = [-7.600902, -6.907755, 0.0, 6.907755, 7.600902]
    result = _col(X_tr, "a")
    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    # Verify symmetry property: arcsinh(-x) = -arcsinh(x)
    np.testing.assert_almost_equal(result[0], -result[4], decimal=10)
    np.testing.assert_almost_equal(result[1], -result[3], decimal=10)


@pytest.mark.parametrize("invalid_scale", [0, -1, -0.5, -100, "string", False])
def test_invalid_scale_raises_error(invalid_scale):
    """Test that non-positive scale values raise ValueError."""
    with pytest.raises(ValueError, match="scale must be a positive number"):
        ArcSinhTransformer(scale=invalid_scale)


@pytest.mark.parametrize("invalid_loc", ["invalid", [1, 2], {"a": 1}, None])
def test_invalid_loc_raises_error(invalid_loc):
    """Test that non-numeric loc values raise ValueError."""
    with pytest.raises(ValueError, match="loc must be a number"):
        ArcSinhTransformer(loc=invalid_loc)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_stores_attributes(make_df):
    """Test that fit stores expected attributes with correct values."""
    X = make_df({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    transformer = ArcSinhTransformer()
    transformer.fit(X)

    assert hasattr(transformer, "variables_")
    assert hasattr(transformer, "feature_names_in_")
    assert hasattr(transformer, "n_features_in_")
    assert transformer.n_features_in_ == 2
    assert set(transformer.variables_) == {"a", "b"}
    assert transformer.feature_names_in_ == ["a", "b"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out(make_df):
    """Test get_feature_names_out returns correct feature names."""
    X = make_df({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    transformer = ArcSinhTransformer()
    transformer.fit(X)

    feature_names = transformer.get_feature_names_out()
    assert feature_names == ["a", "b"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out_with_subset(make_df):
    """Test get_feature_names_out with subset of variables."""
    X = make_df({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    transformer = ArcSinhTransformer(variables=["a"])
    transformer.fit(X)

    feature_names = transformer.get_feature_names_out()
    assert feature_names == ["a", "b", "c"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_behavior_like_log_for_large_values(make_df):
    """Test that arcsinh behaves like log for large positive values."""
    data = {"a": [1000.0, 10000.0, 100000.0]}
    X = make_df(data)
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X)

    log_approx = np.log(2 * np.array(data["a"]))
    np.testing.assert_array_almost_equal(_col(X_tr, "a"), log_approx, decimal=1)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_behavior_like_identity_for_small_values(make_df):
    """Test that arcsinh behaves like identity for small values."""
    data = {"a": [0.001, 0.01, 0.1]}
    X = make_df(data)
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X)

    np.testing.assert_array_almost_equal(
        _col(X_tr, "a"), np.array(data["a"]), decimal=2
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_zero_input_returns_zero(make_df):
    """Test that arcsinh(0) = 0."""
    X = make_df({"a": [0.0]})
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X)

    assert _col(X_tr, "a")[0] == 0.0
