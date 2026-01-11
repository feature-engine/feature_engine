import numpy as np
import pandas as pd
import pytest

from feature_engine.transformation import ArcSinhTransformer


class TestArcSinhTransformer:
    """Test cases for ArcSinhTransformer."""

    def test_default_parameters(self):
        """Test transformer with default parameters."""
        X = pd.DataFrame({"a": [-100, -10, 0, 10, 100], "b": [1, 2, 3, 4, 5]})
        transformer = ArcSinhTransformer()
        X_tr = transformer.fit_transform(X)

        # Check transform was applied
        expected_a = np.arcsinh(X["a"])
        expected_b = np.arcsinh(X["b"])
        np.testing.assert_array_almost_equal(X_tr["a"], expected_a)
        np.testing.assert_array_almost_equal(X_tr["b"], expected_b)

    def test_with_loc_and_scale(self):
        """Test transformer with loc and scale parameters."""
        X = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
        loc = 30.0
        scale = 10.0
        transformer = ArcSinhTransformer(loc=loc, scale=scale)
        X_tr = transformer.fit_transform(X)

        expected = np.arcsinh((X["a"] - loc) / scale)
        np.testing.assert_array_almost_equal(X_tr["a"], expected)

    def test_specific_variables(self):
        """Test transformer with specific variables selected."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        transformer = ArcSinhTransformer(variables=["a", "b"])
        X_tr = transformer.fit_transform(X)

        # Check only specified variables were transformed
        np.testing.assert_array_almost_equal(X_tr["a"], np.arcsinh(X["a"]))
        np.testing.assert_array_almost_equal(X_tr["b"], np.arcsinh(X["b"]))
        # c should be unchanged
        np.testing.assert_array_equal(X_tr["c"], X["c"])

    def test_inverse_transform(self):
        """Test inverse_transform returns original values."""
        X = pd.DataFrame({"a": [-100, -10, 0, 10, 100], "b": [1, 2, 3, 4, 5]})
        X_original = X.copy()
        transformer = ArcSinhTransformer()
        X_tr = transformer.fit_transform(X.copy())
        X_inv = transformer.inverse_transform(X_tr)

        np.testing.assert_array_almost_equal(X_inv["a"], X_original["a"], decimal=10)
        np.testing.assert_array_almost_equal(X_inv["b"], X_original["b"], decimal=10)

    def test_inverse_transform_with_loc_scale(self):
        """Test inverse_transform with loc and scale parameters."""
        X = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
        X_original = X.copy()
        transformer = ArcSinhTransformer(loc=25.0, scale=5.0)
        X_tr = transformer.fit_transform(X.copy())
        X_inv = transformer.inverse_transform(X_tr)

        np.testing.assert_array_almost_equal(X_inv["a"], X_original["a"], decimal=10)

    def test_negative_values(self):
        """Test that transformer handles negative values correctly."""
        X = pd.DataFrame({"a": [-1000, -500, 0, 500, 1000]})
        transformer = ArcSinhTransformer()
        X_tr = transformer.fit_transform(X)

        # arcsinh should handle negative values
        assert X_tr["a"].iloc[0] < 0
        assert X_tr["a"].iloc[1] < 0
        assert X_tr["a"].iloc[2] == 0
        assert X_tr["a"].iloc[3] > 0
        assert X_tr["a"].iloc[4] > 0

    def test_invalid_scale_raises_error(self):
        """Test that invalid scale parameter raises ValueError."""
        with pytest.raises(ValueError, match="scale must be a positive number"):
            ArcSinhTransformer(scale=0)

        with pytest.raises(ValueError, match="scale must be a positive number"):
            ArcSinhTransformer(scale=-1)

    def test_invalid_loc_raises_error(self):
        """Test that invalid loc parameter raises ValueError."""
        with pytest.raises(ValueError, match="loc must be a number"):
            ArcSinhTransformer(loc="invalid")

    def test_fit_stores_attributes(self):
        """Test that fit stores expected attributes."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        transformer = ArcSinhTransformer()
        transformer.fit(X)

        assert hasattr(transformer, "variables_")
        assert hasattr(transformer, "feature_names_in_")
        assert hasattr(transformer, "n_features_in_")
        assert transformer.n_features_in_ == 2
        assert set(transformer.variables_) == {"a", "b"}

    def test_get_feature_names_out(self):
        """Test get_feature_names_out returns correct feature names."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        transformer = ArcSinhTransformer()
        transformer.fit(X)

        feature_names = transformer.get_feature_names_out()
        assert feature_names == ["a", "b"]

    def test_behavior_like_log_for_large_values(self):
        """Test that arcsinh behaves like log for large positive values."""
        X = pd.DataFrame({"a": [1000, 10000, 100000]})
        transformer = ArcSinhTransformer()
        X_tr = transformer.fit_transform(X.copy())

        # For large x: arcsinh(x) ≈ ln(2x) = ln(2) + ln(x)
        log_approx = np.log(2 * X["a"])
        np.testing.assert_array_almost_equal(X_tr["a"], log_approx, decimal=1)
