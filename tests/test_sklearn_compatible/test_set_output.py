import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engine.creation import CyclicalFeatures, MathFeatures, RelativeFeatures
from feature_engine.datetime import DatetimeFeatures, DatetimeSubtraction
from feature_engine.encoding import OneHotEncoder
from feature_engine.transformation import YeoJohnsonTransformer


def test_pipeline_with_set_output_sklearn_last():

    X, y = load_iris(return_X_y=True, as_frame=True)

    pipeline = make_pipeline(
        YeoJohnsonTransformer(), StandardScaler(), LogisticRegression()
    ).set_output(transform="default")

    pipeline.fit(X, y)

    X_t = pipeline[:-1].transform(X)
    assert isinstance(X_t, np.ndarray)

    pipeline.set_output(transform="pandas")
    X_t = pipeline[:-1].transform(X)

    assert isinstance(X_t, pd.DataFrame)


def test_pipeline_with_set_output_featureengine_last():

    X, y = load_iris(return_X_y=True, as_frame=True)

    pipeline = make_pipeline(
        StandardScaler(), YeoJohnsonTransformer(), LogisticRegression()
    ).set_output(transform="default")

    pipeline.fit(X, y)

    X_t = pipeline[:-1].transform(X)
    pipeline.fit(X, y)
    assert isinstance(X_t, pd.DataFrame)

    pipeline.set_output(transform="pandas")
    pipeline.fit(X, y)

    X_t = pipeline[:-1].transform(X)

    assert isinstance(X_t, pd.DataFrame)


def test_individual_transformer():

    X, y = load_iris(return_X_y=True, as_frame=True)

    transformer = YeoJohnsonTransformer()
    transformer.set_output(transform="default")
    transformer.fit(X)

    X_t = transformer.transform(X)
    assert isinstance(X_t, pd.DataFrame)

    transformer.set_output(transform="pandas")
    X_t = transformer.transform(X)

    assert isinstance(X_t, pd.DataFrame)


transformers = [
    OneHotEncoder(variables="feature_1", ignore_format=True),
    CyclicalFeatures(),
    MathFeatures(variables=["feature_1", "feature_2"], func=["sum", "mean"]),
    RelativeFeatures(variables=["feature_1"], reference=["feature_2"], func=["div"]),
]


@pytest.mark.parametrize("transformer", transformers)
def test_transformers_within_pipeline(transformer):
    X = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline(
        [
            ("trs", transformer),
        ]
    ).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)


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
