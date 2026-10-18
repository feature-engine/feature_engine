import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.feature_selection import f_classif, f_regression, r_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold

from feature_engine.selection import ProbeFeatureSelection


def data():
    rng = np.random.default_rng(24)
    X = pd.DataFrame(rng.normal(size=(160, 3)), columns=["signal", "noise", "other"])
    y = 4 * X.signal + rng.normal(scale=0.2, size=len(X))
    return X, y


@pytest.mark.parametrize(
    "method",
    [
        "correlation",
        "f_classif",
        "f_regression",
        "mutual_info_classif",
        "mutual_info_regression",
    ],
)
def test_statistical_methods_are_reproducible_and_keep_signal(method):
    X, y = data()
    if method.endswith("classif"):
        y = (y > 0).astype(int)
    kwargs = dict(importance_method=method, n_probes=3, random_state=7, cv=3)
    selector = ProbeFeatureSelection(**kwargs).fit(X, y)
    other = clone(selector).fit(X, y)
    pd.testing.assert_series_equal(
        selector.feature_importances_, other.feature_importances_
    )
    assert "signal" not in selector.features_to_drop_

    assert len(selector.feature_importances_) == 6
    pd.testing.assert_frame_equal(
        selector.transform(X), X.drop(columns=selector.features_to_drop_)
    )


@pytest.mark.parametrize(
    "method,scorer",
    [
        ("correlation", r_regression),
        ("f_regression", f_regression),
        ("f_classif", f_classif),
    ],
)
def test_statistical_scores_match_training_fold_reference(method, scorer):
    X, y = data()
    if method == "f_classif":
        y = (y > 0).astype(int)
    cv = KFold(4)
    selector = ProbeFeatureSelection(importance_method=method, cv=cv).fit(X, y)
    augmented = pd.concat([X, selector.probe_features_], axis=1)
    scores = []
    for train, _ in cv.split(X):
        score = scorer(augmented.iloc[train], y.iloc[train])
        scores.append(np.abs(score) if method == "correlation" else score[0])
    np.testing.assert_allclose(selector.feature_importances_, np.mean(scores, axis=0))
    np.testing.assert_allclose(
        selector.feature_importances_std_, np.std(scores, axis=0)
    )


def test_permutation_uses_held_out_folds_and_does_not_fit_original_estimator():
    X, y = data()
    estimator = LinearRegression()
    selector = ProbeFeatureSelection(
        estimator, importance_method="permutation", scoring="r2", cv=3
    ).fit(X, y)
    assert not hasattr(estimator, "coef_")
    assert selector.feature_importances_["signal"] > 1
    assert "signal" not in selector.features_to_drop_

    augmented = pd.concat([X, selector.probe_features_], axis=1)
    scores = []
    for train, test in KFold(3).split(X):
        fitted = clone(estimator).fit(augmented.iloc[train], y.iloc[train])
        scores.append(
            permutation_importance(
                fitted, augmented.iloc[test], y.iloc[test], scoring="r2", random_state=0
            ).importances_mean
        )
    np.testing.assert_allclose(selector.feature_importances_, np.mean(scores, axis=0))


def test_group_splitter_supported():
    X, y = data()
    selector = ProbeFeatureSelection(
        importance_method="correlation",
        cv=GroupKFold(4),
        groups=np.repeat(np.arange(8), 20),
    )
    selector.fit(X, y)
    assert "signal" not in selector.features_to_drop_


@pytest.mark.parametrize("method", ["model", "permutation"])
def test_model_based_methods_require_estimator(method):
    with pytest.raises(ValueError, match="estimator is required"):
        ProbeFeatureSelection(importance_method=method).fit(*data())


def test_invalid_method_is_validated_after_set_params():
    selector = ProbeFeatureSelection().set_params(importance_method="unknown")
    with pytest.raises(ValueError, match="importance_method"):
        selector.fit(*data())


def test_correlation_retains_negative_signal_and_handles_constants():
    X, y = data()
    X["signal"] *= -1
    X["constant"] = 1.0
    selector = ProbeFeatureSelection(importance_method="correlation").fit(X, y)
    assert selector.feature_importances_["signal"] > 0.9
    assert selector.feature_importances_["constant"] == 0
    assert "constant" in selector.features_to_drop_


def test_perfect_anova_predictor_has_finite_aggregated_scores():
    X = pd.DataFrame({"signal": [0.0, 1.0] * 20, "constant": [1.0] * 40})
    selector = ProbeFeatureSelection(importance_method="f_classif", cv=2).fit(
        X, X.signal
    )
    assert np.isfinite(selector.feature_importances_).all()
    assert np.isfinite(selector.feature_importances_std_).all()


@pytest.mark.parametrize("method", ["mutual_info_classif", "mutual_info_regression"])
def test_mutual_information_supports_mixed_probe_distributions(method):
    X, y = data()
    if method.endswith("classif"):
        y = (y > 0).astype(int)
    selector = ProbeFeatureSelection(
        importance_method=method, distribution="all", cv=2
    ).fit(X, y)
    assert len(selector.feature_importances_) == 8
    assert np.isfinite(selector.feature_importances_).all()
