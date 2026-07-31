import pandas as pd
import pytest
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.imputation import (
    MissingIndicator,
    ArbitraryImputer,
    CategoricalImputer,
    DropMissingData,
    EndTailImputer,
    MeanImputer,
    RandomSampleImputer,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from tests.estimator_checks.non_fitted_error_checks import (
    check_raises_non_fitted_error_when_fit_fails,
)

_estimators = [
    MeanImputer(),
    ArbitraryImputer(),
    CategoricalImputer(fill_value=0, ignore_format=True),
    EndTailImputer(),
    MissingIndicator(),
    RandomSampleImputer(),
    DropMissingData(),
]

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

if sklearn_version < parse_version("1.6"):

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)

else:

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(
            estimator=estimator,
            expected_failed_checks=estimator._more_tags()["_xfail_checks"],
        )


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "CategoricalImputer":
        estimator.set_params(ignore_format=False)
    if estimator.__class__.__name__ in ["DropMissingData", "MissingIndicator"]:
        estimator.set_params(missing_only=False)
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(transformer):
    if transformer.__class__.__name__ == "CategoricalImputer":
        transformer.set_params(ignore_format=True)
    if transformer.__class__.__name__ in ["DropMissingData", "MissingIndicator"]:
        transformer.set_params(missing_only=False)

    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [6, 7, 8, 9, 10],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)


@pytest.mark.parametrize("estimator", _estimators)
def test_raises_non_fitted_error_when_error_during_fit(estimator):
    if estimator.__class__.__name__ in ["MeanImputer", "EndTailImputer"]:
        # no numerical variables in the df: fails at variable selection.
        X = pd.DataFrame({"cat1": ["a", "b", "c", "a", "b"]})
    elif estimator.__class__.__name__ == "ArbitraryImputer":
        X = pd.DataFrame({"cat1": ["a", "b", "c", "a", "b"]})
    elif estimator.__class__.__name__ == "CategoricalImputer":
        # equally frequent categories: fails after variables_ would have been
        # selected, inside the "frequent" imputation logic itself.
        estimator = estimator.__class__(imputation_method="frequent")
        X = pd.DataFrame({"cat1": ["a", "a", "b", "b"]})
    elif estimator.__class__.__name__ == "RandomSampleImputer":
        # invalid random_state: fails after variables_/X_ would have been set.
        estimator = RandomSampleImputer(seed="observation", random_state="not_a_col")
        X = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    else:
        # AddMissingIndicator, DropMissingData: no reachable failure point
        # once variables are selected, so fail at input validation instead.
        X = pd.DataFrame()

    check_raises_non_fitted_error_when_fit_fails(estimator, X)
