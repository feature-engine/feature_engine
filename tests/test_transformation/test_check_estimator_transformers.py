import pytest
from sklearn.utils.estimator_checks import check_estimator

from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from feature_engine.transformation import (
    BoxCoxTransformer,
    LogCpTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)

_estimators = [
    BoxCoxTransformer(),
    LogTransformer(),
    LogCpTransformer(),
    PowerTransformer(),
    ReciprocalTransformer(),
    YeoJohnsonTransformer(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators[3:])
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)
