"""
File intended to help understand check_estimator tests for Feature-engine's
prediction module. It is not run as part of the battery of acceptance tests.
Works from sklearn > 1.6.
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from feature_engine._prediction.base_predictor import BaseTargetMeanEstimator
from feature_engine._prediction.target_mean_classifier import TargetMeanClassifier
from feature_engine._prediction.target_mean_regressor import TargetMeanRegressor
from feature_engine.tags import _return_tags

_estimators = [BaseTargetMeanEstimator(), TargetMeanClassifier(), TargetMeanRegressor()]

FAILED_CHECKS = _return_tags()["_xfail_checks"]

EXPECTED_FAILED_CHECKS = {
    "BaseTargetMeanEstimator": FAILED_CHECKS,
    "TargetMeanClassifier": FAILED_CHECKS,
    "TargetMeanRegressor": FAILED_CHECKS,
}


@parametrize_with_checks(
    estimators=_estimators,
    expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS.get(
        est.__class__.__name__, {}
    ),
)
def test_sklearn_compatible_creator(estimator, check):
    check(estimator)
