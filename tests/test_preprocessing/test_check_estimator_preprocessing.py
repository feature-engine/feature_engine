import pytest

from numpy import nan
from sklearn import clone
from sklearn.exceptions import NotFittedError
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
    X.loc[len(X)-1] = nan
    transformer = clone(estimator)

    with pytest.raises(ValueError):
        transformer.fit(X, y)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        transformer.transform(X)
