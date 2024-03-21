import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OrdinalEncoder as sk_OrdinalEncoder

from feature_engine.encoding import OrdinalEncoder
from feature_engine.imputation import DropMissingData
from feature_engine.pipeline import Pipeline


@pytest.fixture(scope="module")
def create_data():
    X = pd.DataFrame(
        dict(
            x1=[2, 1, 1, 0, np.nan],
            x2=["a", np.nan, "b", np.nan, "a"],
        )
    )
    y = pd.Series([1, 2, 3, 4, 5])
    return X, y


def test_pipeline_with_1_transformer(create_data):
    X, y = create_data
    dmd = DropMissingData().fit(X)
    Xt1, yt1 = dmd.transform_x_y(X=X, y=y)

    # transform_x_y
    pipe = Pipeline([("drop", DropMissingData())])
    pipe.fit(X, y)
    Xt, yt = pipe.transform_x_y(X, y)
    pd.testing.assert_frame_equal(Xt1, Xt)
    pd.testing.assert_series_equal(yt1, yt)

    # transform
    Xt = pipe.transform(X)
    pd.testing.assert_frame_equal(Xt1, Xt)

    # fit_transform
    pipe = Pipeline([("drop", DropMissingData())])
    Xt = pipe.fit_transform(X=X, y=y)
    pd.testing.assert_frame_equal(Xt1, Xt)
    pd.testing.assert_series_equal(yt1, yt)


def test_with_2_transformers_from_fe(create_data):
    X, y = create_data
    dmd = DropMissingData().fit(X)
    Xt1, yt1 = dmd.transform_x_y(X=X, y=y)
    Xt1 = OrdinalEncoder(encoding_method="arbitrary").fit_transform(Xt1, yt1)

    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", OrdinalEncoder(encoding_method="arbitrary")),
        ]
    )

    # transform_x_y
    pipe.fit(X, y)
    Xt, yt = pipe.transform_x_y(X, y)
    pd.testing.assert_frame_equal(Xt1, Xt)
    pd.testing.assert_series_equal(yt1, yt)

    # transform
    Xt = pipe.transform(X)
    pd.testing.assert_frame_equal(Xt1, Xt)

    # fit_transform
    Xt, yt = pipe.transform_x_y(X, y)
    pd.testing.assert_frame_equal(Xt1, Xt)
    pd.testing.assert_series_equal(yt1, yt)


def test_pipeline_with_estimator(create_data):
    X, y = create_data
    dmd = DropMissingData().fit(X)
    Xt1, yt1 = dmd.transform_x_y(X, y)
    Xt1 = OrdinalEncoder(encoding_method="arbitrary").fit_transform(Xt1)
    model = Lasso(random_state=10).fit(Xt1, yt1)
    preds = model.predict(Xt1)

    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", OrdinalEncoder(encoding_method="arbitrary")),
            ("lasso", Lasso(random_state=10)),
        ]
    )
    # predict
    pipe.fit(X, y)
    preds_pipe = pipe.predict(X)
    assert (preds == preds_pipe).all()


def test_with_1_transformers_from_sklearn(create_data):
    X, y = create_data
    dmd = DropMissingData().fit(X)
    Xt1, yt1 = dmd.transform_x_y(X=X, y=y)
    Xt1 = sk_OrdinalEncoder().set_output(transform="pandas").fit_transform(Xt1)

    # pipeline of transformers only
    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", sk_OrdinalEncoder()),
        ]
    ).set_output(transform="pandas")

    # transform_x_y
    pipe.fit(X, y)
    Xt, yt = pipe.transform_x_y(X, y)
    pd.testing.assert_frame_equal(Xt, Xt1)
    pd.testing.assert_series_equal(yt, yt1)

    # transform
    Xt = pipe.transform(X)
    pd.testing.assert_frame_equal(Xt1, Xt)

    # fit_transform
    Xt = pipe.fit_transform(X, y)
    pd.testing.assert_frame_equal(Xt1, Xt)


def test_with_trasnformer_and_estimator_from_sklearn(create_data):
    X, y = create_data
    dmd = DropMissingData().fit(X)
    Xt1, yt1 = dmd.transform_x_y(X=X, y=y)
    Xt1 = sk_OrdinalEncoder().set_output(transform="pandas").fit_transform(Xt1)
    model = Lasso(random_state=10).fit(Xt1, yt1)
    preds = model.predict(Xt1)

    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", sk_OrdinalEncoder()),
            ("lasso", Lasso(random_state=10)),
        ]
    ).set_output(transform="pandas")
    pipe.fit(X, y)
    preds_pipe = pipe.predict(X)
    assert (preds_pipe == preds).all()

    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", sk_OrdinalEncoder()),
            ("lasso", Lasso(random_state=10)),
        ]
    )
    pipe.fit(X, y)
    preds_pipe = pipe.predict(X)
    assert (preds_pipe == preds).all()
