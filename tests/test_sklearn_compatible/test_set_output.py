import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from feature_engine.transformation import YeoJohnsonTransformer


def test_pipeline_with_set_output_sklearn_last():

    X, y = load_iris(return_X_y=True, as_frame=True)

    pipeline = make_pipeline(
        YeoJohnsonTransformer(), StandardScaler(), LogisticRegression()
    ).set_output(transform="default")
    pipeline.fit(X, y)

    X_t = pipeline[:-1].fit_transform(X, y)
    assert isinstance(X_t, np.ndarray)

    pipeline.set_output(transform="pandas")
    X_t = pipeline[:-1].fit_transform(X, y)

    assert isinstance(X_t, pd.DataFrame)


def test_pipeline_with_set_output_featureengine_last():

    X, y = load_iris(return_X_y=True, as_frame=True)

    pipeline = make_pipeline(
        StandardScaler(), YeoJohnsonTransformer(), LogisticRegression()
    ).set_output(transform="default")
    pipeline.fit(X, y)

    X_t = pipeline[:-1].fit_transform(X, y)
    assert isinstance(X_t, pd.DataFrame)

    pipeline.set_output(transform="pandas")
    X_t = pipeline[:-1].fit_transform(X, y)

    assert isinstance(X_t, pd.DataFrame)
