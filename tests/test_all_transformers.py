import pytest

from sklearn.utils.estimator_checks import check_estimator

from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.transformation import (
    BoxCoxTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)


def test_sklearn_trainsformer_wrapper():
    check_estimator(SklearnTransformerWrapper())


@pytest.mark.parametrize(
    "Estimator", [BoxCoxTransformer(), LogTransformer(), PowerTransformer(),
                  ReciprocalTransformer(), YeoJohnsonTransformer(),
                  ]
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
