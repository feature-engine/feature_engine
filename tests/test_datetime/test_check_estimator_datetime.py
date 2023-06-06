import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.datetime import DatetimeFeatures, DatetimeSubtraction
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [DatetimeFeatures()]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


transformers = [
    DatetimeFeatures(),
    DatetimeSubtraction(variables="feature_1", reference="feature_2"),
]


@pytest.mark.parametrize("transformer", transformers)
def test_datetime_transformers(transformer):
    X = pd.DataFrame(
        {
            "feature_1": [
                "2014-05-05",
                "2014-05-05",
                "2014-05-05",
                "2014-05-05",
                "2014-05-05",
            ],
            "feature_2": [
                "2014-05-05",
                "2014-05-05",
                "2014-05-05",
                "2014-05-05",
                "2014-05-05",
            ],
        },
    )
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline(
        [
            ("trs", transformer),
        ]
    ).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)
