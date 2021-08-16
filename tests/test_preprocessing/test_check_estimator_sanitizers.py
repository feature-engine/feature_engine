import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.preprocessing import (
    MatchColumnsToTrainSet
)


@pytest.mark.parametrize(
    "Estimator",
    [
        MatchColumnsToTrainSet(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
