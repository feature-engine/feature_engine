import numpy as np
import pandas as pd
import pytest

from feature_engine.transformation import ArcSinhTransformer


@pytest.fixture
def df_numerical():
    """Fixture providing sample numerical data with positive and negative values."""
    return pd.DataFrame({
        "a": [-100, -10, 0, 10, 100],
        "b": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def df_multi_column():
    """Fixture providing DataFrame with multiple columns."""
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
    })


def test_default_parameters(df_numerical):
    """Test transformer with default parameters applies arcsinh to all columns."""
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(df_numerical.copy())

    expected_a = np.arcsinh(df_numerical["a"])
    expected_b = np.arcsinh(df_numerical["b"])
    np.testing.assert_array_almost_equal(X_tr["a"], expected_a)
    np.testing.assert_array_almost_equal(X_tr["b"], expected_b)


def test_specific_variables(df_multi_column):
    """Test transformer with specific variables selected."""
    transformer = ArcSinhTransformer(variables=["a", "b"])
    X_tr = transformer.fit_transform(df_multi_column.copy())

    np.testing.assert_array_almost_equal(
        X_tr["a"], np.arcsinh(df_multi_column["a"])
    )
    np.testing.assert_array_almost_equal(
        X_tr["b"], np.arcsinh(df_multi_column["b"])
    )
    np.testing.assert_array_equal(X_tr["c"], df_multi_column["c"])


def test_with_loc_and_scale():
    """Test transformer with loc and scale parameters."""
    X = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
    loc = 30.0
    scale = 10.0
    transformer = ArcSinhTransformer(loc=loc, scale=scale)
    X_tr = transformer.fit_transform(X.copy())

    expected = np.arcsinh((X["a"] - loc) / scale)
    np.testing.assert_array_almost_equal(X_tr["a"], expected)
    np.testing.assert_almost_equal(X_tr["a"].iloc[2], 0.0, decimal=10)


@pytest.mark.parametrize("loc", [0.0, 10.0, -10.0, 100.5])
def test_various_loc_values(loc):
    """Test that various loc values work correctly."""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    transformer = ArcSinhTransformer(loc=loc)
    X_tr = transformer.fit_transform(X.copy())

    expected = np.arcsinh((X["a"] - loc) / 1.0)
    np.testing.assert_array_almost_equal(X_tr["a"], expected)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 10.0, 100.0])
def test_various_scale_values(scale):
    """Test that various scale values work correctly."""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    transformer = ArcSinhTransformer(scale=scale)
    X_tr = transformer.fit_transform(X.copy())

    expected = np.arcsinh((X["a"] - 0.0) / scale)
    np.testing.assert_array_almost_equal(X_tr["a"], expected)


def test_inverse_transform(df_numerical):
    """Test inverse_transform returns original values."""
    X_original = df_numerical.copy()
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(df_numerical.copy())
    X_inv = transformer.inverse_transform(X_tr)

    np.testing.assert_array_almost_equal(X_inv["a"], X_original["a"], decimal=10)
    np.testing.assert_array_almost_equal(X_inv["b"], X_original["b"], decimal=10)


def test_inverse_transform_with_loc_scale():
    """Test inverse_transform with loc and scale parameters."""
    X = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
    X_original = X.copy()
    transformer = ArcSinhTransformer(loc=25.0, scale=5.0)
    X_tr = transformer.fit_transform(X.copy())
    X_inv = transformer.inverse_transform(X_tr)

    np.testing.assert_array_almost_equal(X_inv["a"], X_original["a"], decimal=10)


def test_negative_values():
    """Test that transformer handles negative values correctly."""
    X = pd.DataFrame({"a": [-1000, -500, 0, 500, 1000]})
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X.copy())

    # Expected values: arcsinh([ -1000, -500, 0, 500, 1000 ])
    expected = [-7.600902, -6.907755, 0.0, 6.907755, 7.600902]
    np.testing.assert_array_almost_equal(X_tr["a"], expected, decimal=5)

    # Verify symmetry property: arcsinh(-x) = -arcsinh(x)
    np.testing.assert_almost_equal(
        X_tr["a"].iloc[0], -X_tr["a"].iloc[4], decimal=10
    )
    np.testing.assert_almost_equal(
        X_tr["a"].iloc[1], -X_tr["a"].iloc[3], decimal=10
    )


@pytest.mark.parametrize("invalid_scale", [0, -1, -0.5, -100])
def test_invalid_scale_raises_error(invalid_scale):
    """Test that non-positive scale values raise ValueError."""
    with pytest.raises(ValueError, match="scale must be a positive number"):
        ArcSinhTransformer(scale=invalid_scale)


@pytest.mark.parametrize("invalid_loc", ["invalid", [1, 2], {"a": 1}, None])
def test_invalid_loc_raises_error(invalid_loc):
    """Test that non-numeric loc values raise ValueError."""
    with pytest.raises(ValueError, match="loc must be a number"):
        ArcSinhTransformer(loc=invalid_loc)


def test_fit_stores_attributes():
    """Test that fit stores expected attributes with correct values."""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    transformer = ArcSinhTransformer()
    transformer.fit(X)

    assert hasattr(transformer, "variables_")
    assert hasattr(transformer, "feature_names_in_")
    assert hasattr(transformer, "n_features_in_")
    assert transformer.n_features_in_ == 2
    assert set(transformer.variables_) == {"a", "b"}
    assert transformer.feature_names_in_ == ["a", "b"]


def test_get_feature_names_out():
    """Test get_feature_names_out returns correct feature names."""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    transformer = ArcSinhTransformer()
    transformer.fit(X)

    feature_names = transformer.get_feature_names_out()
    assert feature_names == ["a", "b"]


def test_get_feature_names_out_with_subset():
    """Test get_feature_names_out with subset of variables."""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    transformer = ArcSinhTransformer(variables=["a"])
    transformer.fit(X)

    feature_names = transformer.get_feature_names_out()
    assert feature_names == ["a", "b", "c"]


def test_behavior_like_log_for_large_values():
    """Test that arcsinh behaves like log for large positive values."""
    X = pd.DataFrame({"a": [1000, 10000, 100000]})
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X.copy())

    log_approx = np.log(2 * X["a"])
    np.testing.assert_array_almost_equal(X_tr["a"], log_approx, decimal=1)


def test_behavior_like_identity_for_small_values():
    """Test that arcsinh behaves like identity for small values."""
    X = pd.DataFrame({"a": [0.001, 0.01, 0.1]})
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X.copy())

    np.testing.assert_array_almost_equal(X_tr["a"], X["a"], decimal=2)


def test_zero_input_returns_zero():
    """Test that arcsinh(0) = 0."""
    X = pd.DataFrame({"a": [0.0]})
    transformer = ArcSinhTransformer()
    X_tr = transformer.fit_transform(X.copy())

    assert X_tr["a"].iloc[0] == 0.0
