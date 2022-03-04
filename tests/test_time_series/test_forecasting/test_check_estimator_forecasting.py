import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.estimator_checks import check_feature_engine_estimator
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures

_estimators = [
    LagFeatures(missing_values="ignore"),
    WindowFeatures(missing_values="ignore"),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_sort_index(df_time, estimator):
    X = df_time.copy()

    # Shuffle dataframe
    Xs = X.sample(len(df_time)).copy()

    transformer = clone(estimator)
    X_tr = transformer.fit_transform(Xs)

    A = X[transformer.variables_].iloc[0:4].values
    B = X_tr[transformer.get_feature_names_out(transformer.variables_)].iloc[1:5].values
    assert (A == B).all()

    transformer = LagFeatures(sort_index=False)
    X_tr = transformer.fit_transform(Xs)

    A = Xs[transformer.variables_].iloc[0:4].values
    B = X_tr[transformer.get_feature_names_out(transformer.variables_)].iloc[1:5].values
    assert (A == B).all()


@pytest.mark.parametrize("estimator", _estimators)
def test_error_when_not_unique_values_in_index(df_time, estimator):
    X = df_time.copy()

    # introduce dupes in index
    tmp = X.head(2).copy()
    tmp.iloc[0] = [1, 1, 1, "blue"]
    Xd = pd.concat([X, tmp], axis=0)

    transformer = clone(estimator)

    with pytest.raises(NotImplementedError):
        transformer.fit(Xd)

    transformer.fit(X)
    with pytest.raises(NotImplementedError):
        transformer.transform(Xd)


@pytest.mark.parametrize("estimator", _estimators)
def test_error_when_nan_in_index(df_time, estimator):
    X = df_time.copy()

    # Introduce NaN in index.
    tmp = X.head(1).copy()
    tmp.index = [np.nan]
    Xd = pd.concat([X, tmp], axis=0)

    transformer = clone(estimator)

    with pytest.raises(NotImplementedError):
        transformer.fit(Xd)

    transformer.fit(X)
    with pytest.raises(NotImplementedError):
        transformer.transform(Xd)
