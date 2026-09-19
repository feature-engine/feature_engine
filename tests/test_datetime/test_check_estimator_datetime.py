import datetime

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.datetime import (
    DatetimeFeatures,
    DatetimeOrdinal,
    DatetimeSubtraction,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from tests.estimator_checks.fit_functionality_checks import (
    check_transform_returns_training_variable_order,
)

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


@pytest.mark.parametrize(
    "estimator",
    [
        DatetimeFeatures(variables=["date_1"], features_to_extract=["month"]),
        DatetimeSubtraction(variables="date_1", reference="date_2"),
        DatetimeOrdinal(variables=["date_1"]),
    ],
)
def test_transform_returns_training_variable_order(estimator, make_df):
    data = {
        "date_1": [datetime.datetime(2020, month, 1) for month in range(1, 7)],
        "var_A": [1, 2, 3, 4, 5, 6],
        "date_2": [datetime.datetime(2021, month, 15) for month in range(1, 7)],
    }
    check_transform_returns_training_variable_order(
        estimator, make_df, data, [0, 1, 0, 1, 1, 0]
    )
