import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.transformation import (
    BoxCoxTransformer,
    LogCpTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)


@pytest.mark.parametrize(
    "Estimator",
    [
        BoxCoxTransformer(),
        LogTransformer(),
        LogCpTransformer(),
        PowerTransformer(),
        ReciprocalTransformer(),
        YeoJohnsonTransformer(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
