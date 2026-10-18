import numpy as np
import pandas as pd
import pytest

from feature_engine.selection import DropHighPSIFeatures


@pytest.mark.parametrize("strategy", ["equal_width", "equal_frequency"])
@pytest.mark.parametrize("categorical", [False, True])
def test_missingness_drift_has_its_own_bin(strategy, categorical):
    basis = [1.0, 1.0, 2.0, 2.0, np.nan, np.nan, np.nan, np.nan]
    test = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, np.nan, np.nan]
    X = pd.DataFrame({"x": basis + test})
    if categorical:
        X["x"] = X["x"].map({1.0: "a", 2.0: "b"})
    selector = DropHighPSIFeatures(
        variables="all", missing_values="include", bins=2, strategy=strategy
    ).fit(X)
    expected = 0.25 * np.log(0.5 / 0.25) + 2 * 0.125 * np.log(0.375 / 0.25)
    assert selector.psi_values_["x"] == pytest.approx(expected)
    ignored = DropHighPSIFeatures(
        variables="all", missing_values="ignore", bins=2, strategy=strategy
    ).fit(X)
    assert ignored.psi_values_["x"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "basis,test",
    [
        ([np.nan] * 4, [np.nan] * 4),
        ([np.nan] * 4, [1.0, 2.0, 3.0, 4.0]),
        ([1.0, 2.0, 3.0, 4.0], [np.nan] * 4),
    ],
)
@pytest.mark.parametrize("threshold", [0.25, "auto"])
def test_all_missing_partition_is_supported(basis, test, threshold):
    selector = DropHighPSIFeatures(
        missing_values="include", bins=2, threshold=threshold
    ).fit(pd.DataFrame({"x": basis + test}))
    assert np.isfinite(selector.psi_values_["x"])
    if all(pd.isna(basis)) and all(pd.isna(test)):
        assert selector.psi_values_["x"] == 0
        assert selector.features_to_drop_ == []
    else:
        assert selector.features_to_drop_ == ["x"]


def test_auto_threshold_counts_missing_category():
    X = pd.DataFrame({"x": ["a", None, "b", None, "a", "a", "b", None]})
    selector = DropHighPSIFeatures(
        missing_values="include", variables="all", bins=2, threshold="auto"
    ).fit(X)
    expected_threshold = selector._calculate_auto_threshold(4, 4, 3)
    assert ("x" in selector.features_to_drop_) == (
        selector.psi_values_["x"] > expected_threshold
    )


def test_missing_bin_does_not_collide_with_real_category():
    X = pd.DataFrame({"x": ["missing", None] * 4})
    selector = DropHighPSIFeatures(missing_values="include", variables="all", bins=2)
    left, right = selector._observation_frequency_per_bin(X.iloc[:4], X.iloc[4:])
    assert len(left) == len(right) == 2
    assert left.sum() == right.sum() == 1


def test_duplicate_row_labels_do_not_duplicate_observations():
    X = pd.DataFrame(
        {
            "period": [0] * 8 + [1] * 8,
            "x": [
                1.0,
                1.0,
                2.0,
                2.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                np.nan,
                np.nan,
            ],
        },
        index=[0] * 16,
    )
    selector = DropHighPSIFeatures(
        split_col="period", cut_off=0, missing_values="include", bins=2
    ).fit(X)
    expected = 0.25 * np.log(2) + 0.25 * np.log(1.5)
    assert selector.psi_values_["x"] == pytest.approx(expected)


def test_infinite_values_are_not_treated_as_missing():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.inf] * 2})
    with pytest.raises(ValueError, match="inf"):
        DropHighPSIFeatures(missing_values="include", bins=2).fit(X)


def test_all_missing_categorical_feature_has_zero_psi():
    X = pd.DataFrame({"x": pd.Series([None] * 8, dtype="object")})
    selector = DropHighPSIFeatures(
        missing_values="include", variables="all", bins=2, threshold="auto"
    ).fit(X)
    assert selector.psi_values_["x"] == 0
    assert selector.features_to_drop_ == []
