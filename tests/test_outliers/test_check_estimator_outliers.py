import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.outliers import ArbitraryOutlierCapper, OutlierTrimmer, Winsorizer


@pytest.mark.parametrize(
    "Estimator",
    [
        ArbitraryOutlierCapper(max_capping_dict={"0": 10}),
        OutlierTrimmer(),
        Winsorizer(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
