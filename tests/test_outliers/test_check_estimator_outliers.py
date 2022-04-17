import pytest
from sklearn.utils.estimator_checks import check_estimator

from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from feature_engine.outliers import ArbitraryOutlierCapper, OutlierTrimmer, Winsorizer

_estimators = [
    ArbitraryOutlierCapper(max_capping_dict={"0": 10}),
    OutlierTrimmer(),
    Winsorizer(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "ArbitraryOutlierCapper":
        estimator.set_params(max_capping_dict={"var_1": 10})
    return check_feature_engine_estimator(estimator)
