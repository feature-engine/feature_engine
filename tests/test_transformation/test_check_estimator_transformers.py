import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.transformation import (
    ArcsinTransformer,
    BoxCoxTransformer,
    LogCpTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [
    BoxCoxTransformer(),
    LogTransformer(),
    LogCpTransformer(),
    ArcsinTransformer(),
    PowerTransformer(),
    ReciprocalTransformer(),
    YeoJohnsonTransformer(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators[4:])
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(transformer):
    X = pd.DataFrame(
        {"feature_1": [0.1, 0.2, 0.3, 0.4, 0.5], "feature_2": [0.6, 0.7, 0.8, 0.9, 0.1]}
    )
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)
