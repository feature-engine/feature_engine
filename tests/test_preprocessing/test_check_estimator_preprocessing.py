import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.preprocessing import MatchVariables


@pytest.mark.parametrize(
    "Estimator",
    [
        MatchVariables(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
