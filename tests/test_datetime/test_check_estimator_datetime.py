import pytest

from feature_engine.datetime import DatetimeFeatures, DatetimeSubtraction
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [DatetimeFeatures()]#, DatetimeSubtraction(variables=["var_1", "var_2"], reference=["var_3"])]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)
