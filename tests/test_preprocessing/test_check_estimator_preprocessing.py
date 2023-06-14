import pandas as pd
import pytest
from numpy import nan
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.preprocessing import MatchCategories, MatchVariables
from tests.estimator_checks.estimator_checks import (
    check_feature_engine_estimator,
    test_df,
)

_estimators = [MatchCategories(ignore_format=True), MatchVariables()]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", [MatchCategories(), MatchVariables()])
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_raises_non_fitted_error_when_error_during_fit(estimator):
    X, y = test_df(categorical=True)
    X.loc[len(X) - 1] = nan
    transformer = clone(estimator)

    with pytest.raises(ValueError):
        transformer.fit(X, y)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        transformer.transform(X)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(transformer):
    X = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)
