import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.selection import (
    MRMR,
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropFeatures,
    DropHighPSIFeatures,
    ProbeFeatureSelection,
    RecursiveFeatureAddition,
    RecursiveFeatureElimination,
    SelectByInformationValue,
    SelectByShuffling,
    SelectBySingleFeaturePerformance,
    SelectByTargetMeanPerformance,
    SmartCorrelatedSelection,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from tests.estimator_checks.init_params_triggered_functionality_checks import (
    check_confirm_variables,
    check_raises_error_if_only_1_variable,
)

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

_logreg = LogisticRegression(C=0.0001, max_iter=2, random_state=1)

_estimators = [
    DropFeatures(features_to_drop=["x0"]),
    DropConstantFeatures(missing_values="ignore"),
    DropDuplicateFeatures(),
    DropCorrelatedFeatures(),
    DropHighPSIFeatures(bins=5),
    SmartCorrelatedSelection(),
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectByTargetMeanPerformance(bins=3, regression=False),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy", threshold=-100),
    SelectByInformationValue(bins=2),
    ProbeFeatureSelection(estimator=_logreg, scoring="accuracy"),
    MRMR(regression=False),
]

_multivariate_estimators = [
    DropDuplicateFeatures(),
    DropCorrelatedFeatures(),
    SmartCorrelatedSelection(),
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy", threshold=-100),
    MRMR(regression=False),
]

_univariate_estimators = [
    DropFeatures(features_to_drop=["var_1"]),
    DropConstantFeatures(missing_values="ignore"),
    DropHighPSIFeatures(bins=5),
    SelectByTargetMeanPerformance(bins=3, regression=False, threshold=0),
    SelectBySingleFeaturePerformance(
        estimator=_logreg,
        scoring="accuracy",
        threshold=0,
    ),
    SelectByInformationValue(bins=2),
    ProbeFeatureSelection(estimator=_logreg, scoring="accuracy"),
]

_model_based_estimators = [
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy", threshold=-100),
    ProbeFeatureSelection(estimator=_logreg, scoring="accuracy"),
]

if sklearn_version < parse_version("1.6"):

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)

else:
    # In sklearn 1.6. the API changes break the tests for the target mean selector.
    # We need to investigate further.
    # TODO: investigate checks for target mean selector.
    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        if estimator.__class__.__name__ != "SelectByTargetMeanPerformance":
            failed_tests = estimator._more_tags()["_xfail_checks"]
            return check_estimator(
                estimator=estimator, expected_failed_checks=failed_tests
            )


@pytest.mark.parametrize("estimator", _univariate_estimators)
def test_check_univariate_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("estimator", _multivariate_estimators)
def test_check_multivariate_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator, needs_group=True)


@pytest.mark.parametrize("estimator", _estimators)
def test_confirm_variables(estimator):
    if estimator.__class__.__name__ != "DropFeatures":
        return check_confirm_variables(estimator)


@pytest.mark.parametrize("estimator", _multivariate_estimators)
def test_raises_error_if_only_1_variable(estimator):
    return check_raises_error_if_only_1_variable(estimator)


@pytest.mark.parametrize("estimator", _model_based_estimators)
def test_raises_error_when_no_estimator_passed(estimator):
    # these selectors need an estimator as an input param
    # test error otherwise.
    with pytest.raises(TypeError):
        estimator()


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(transformer):
    if transformer.__class__.__name__ == "DropFeatures":
        transformer.set_params(features_to_drop=["feature_1"])

    if transformer.__class__.__name__ == "ProbeFeatureSelection":
        transformer.set_params(cv=2)

    if transformer.__class__.__name__ == "DropHighPSIFeatures":
        transformer.set_params(bins=2)

    X = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X, y)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)
