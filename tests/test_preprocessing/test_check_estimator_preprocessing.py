import pandas as pd
import pytest
import sklearn
from numpy import nan
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.preprocessing import MatchCategories, MatchVariables
from feature_engine.tags import _return_tags
from tests.estimator_checks.estimator_checks import (
    check_feature_engine_estimator,
    test_df,
)

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

_estimators = [MatchCategories(ignore_format=True), MatchVariables()]

if sklearn_version < parse_version("1.6"):

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)

else:
    FAILED_CHECKS = _return_tags()["_xfail_checks"]
    FAILED_CHECKS_MATCHCOLS = _return_tags()["_xfail_checks"]

    msg1 = "input shape of dataframes in fit and transform can differ"
    msg2 = (
        "transformer takes categorical variables, and inf cannot be determined"
        "on these variables. Thus, check is not implemented"
    )

    FAILED_CHECKS.update({"check_estimators_nan_inf": msg2})
    FAILED_CHECKS_MATCHCOLS.update(
        {
            "check_transformer_general": msg1,
            "check_estimators_nan_inf": msg2,
        }
    )

    @pytest.mark.parametrize(
        "estimator, failed_tests",
        [
            (_estimators[0], FAILED_CHECKS),
            (_estimators[1], FAILED_CHECKS_MATCHCOLS),
        ],
    )
    def test_check_estimator_from_sklearn(estimator, failed_tests):
        return check_estimator(estimator=estimator, expected_failed_checks=failed_tests)


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
