import pytest
from sklearn.utils.estimator_checks import check_estimator

from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from feature_engine.preprocessing import MatchVariables

_estimators = [MatchVariables()]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)
