import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.datetime import (
    DatetimeFeatures,
    DatetimeOrdinal,
    DatetimeSubtraction,
)
from tests.backend_helpers import frame_to_dict
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


_REORDER_DATA = {
    "date_1": ["2014-05-05", "2015-06-10", "2016-07-15"],
    "date_2": ["2013-01-01", "2014-02-02", "2015-03-03"],
    "num": [1, 2, 3],
}


@pytest.mark.parametrize(
    "transformer",
    [
        DatetimeFeatures(drop_original=False),
        DatetimeOrdinal(drop_original=False),
        DatetimeSubtraction(variables="date_1", reference="date_2"),
    ],
)
def test_transform_reorders_columns_as_in_train_set(make_df, transformer):
    X = make_df(_REORDER_DATA)
    expected = transformer.fit(X).transform(X)

    reordered = ["num", "date_2", "date_1"]
    X_reordered = make_df({var: _REORDER_DATA[var] for var in reordered})
    Xt = transformer.transform(X_reordered)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == list(expected.columns)
    assert frame_to_dict(Xt) == frame_to_dict(expected)
