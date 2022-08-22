import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.transformation import (
    ArcsinTransformer,
    BoxCoxTransformer,
    LogCpTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [
    BoxCoxTransformer(),
    LogTransformer(),
    LogCpTransformer(),
    ArcsinTransformer(),
    PowerTransformer(),
    ReciprocalTransformer(),
    YeoJohnsonTransformer(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators[4:])
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)
