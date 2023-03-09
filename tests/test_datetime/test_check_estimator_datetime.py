import pytest

from feature_engine.datetime import DatetimeFeatures, DatetimeSubtraction
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [DatetimeFeatures(), DatetimeSubtraction()]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)
