import re

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from feature_engine._base_transformers.mixins import TransformXyMixin

BACKENDS = [(pd.DataFrame, pd.Series), (pl.DataFrame, pl.Series)]


class MockTransformer(TransformXyMixin, BaseEstimator):
    def fit(self, X):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        # reorders columns as in fit, as transformers do, and drops rows at
        # positions 2 and 4, backend-agnostic
        nw_X = nw.from_native(X, eager_only=True).select(self.feature_names_in_)
        keep = [i for i in range(len(nw_X)) if i not in (2, 4)]
        return nw_X[keep].to_native()


@pytest.mark.parametrize("make_df, make_series", BACKENDS)
def test_transform_x_y_single_target(make_df, make_series):
    X = make_df({"a": [0, 1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14, 15]})
    y = make_series([0, 1, 2, 3, 4, 5])
    transformer = MockTransformer().fit(X)

    Xt, yt = transformer.transform_x_y(X, y)

    assert len(Xt) == 4
    assert len(yt) == 4
    assert nw.from_native(yt, series_only=True).to_list() == [0, 1, 3, 5]
    assert transformer.n_features_in_ == 2


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_x_y_multioutput_target(make_df):
    X = make_df({"a": [0, 1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14, 15]})
    y = make_df({"t1": [0, 1, 2, 3, 4, 5], "t2": [0, 10, 20, 30, 40, 50]})
    transformer = MockTransformer().fit(X)

    Xt, yt = transformer.transform_x_y(X, y)

    assert len(Xt) == 4
    assert len(yt) == 4
    nw_yt = nw.from_native(yt, eager_only=True)
    assert nw_yt["t1"].to_list() == [0, 1, 3, 5]
    assert nw_yt["t2"].to_list() == [0, 10, 30, 50]


def test_transform_x_y_pandas_index_alignment(df_vartypes):
    # pandas branch keeps the original (non-default) index aligned between X and y
    class DropFirstAndLast(TransformXyMixin, BaseEstimator):
        def fit(self, X):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = list(X.columns)
            return self

        def transform(self, X):
            return X.iloc[1:-1].copy()

    y = pd.Series(range(len(df_vartypes)), index=df_vartypes.index)
    transformer = DropFirstAndLast().fit(df_vartypes)
    Xt, yt = transformer.transform_x_y(df_vartypes, y)

    assert len(Xt) == len(yt)
    assert len(Xt) != len(df_vartypes)
    assert (Xt.index == yt.index).all()
    assert (Xt.index == [1, 2]).all()


def test_transform_x_y_with_duplicated_pandas_index():
    X = pd.DataFrame({"a": range(6), "b": range(10, 16)}, index=[0, 0, 1, 1, 2, 2])
    y = pd.Series(range(6), index=X.index)

    Xt, yt = MockTransformer().fit(X).transform_x_y(X, y)

    pd.testing.assert_frame_equal(Xt, X.iloc[[0, 1, 3, 5]])
    pd.testing.assert_series_equal(yt, y.iloc[[0, 1, 3, 5]])


@pytest.mark.parametrize("make_df, make_series", BACKENDS)
def test_transform_x_y_when_transform_removes_all_rows(make_df, make_series):
    class DropAllRows(MockTransformer):
        def transform(self, X):
            nw_X = nw.from_native(X, eager_only=True).select(self.feature_names_in_)
            return nw_X.head(0).to_native()

    X = make_df({"a": [0, 1], "b": [10, 11]})
    y = make_series([0, 1])

    Xt, yt = DropAllRows().fit(X).transform_x_y(X, y)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["a", "b"]
    assert len(Xt) == 0
    assert len(yt) == 0


@pytest.mark.parametrize("make_df, make_series", BACKENDS)
def test_transform_x_y_raises_error_if_not_fitted(make_df, make_series):
    X = make_df({"a": [0, 1, 2], "b": [10, 11, 12]})
    y = make_series([0, 1, 2])
    msg = (
        "This MockTransformer instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        MockTransformer().transform_x_y(X, y)
