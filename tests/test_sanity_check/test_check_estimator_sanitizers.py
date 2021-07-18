import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.sanity_check import (
    SimilarColumns
)


@pytest.mark.parametrize(
    "Estimator",
    [
        SimilarColumns(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
