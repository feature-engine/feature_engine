import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.selection import (
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropFeatures,
    DropHighPSIFeatures,
    RecursiveFeatureAddition,
    RecursiveFeatureElimination,
    SelectByShuffling,
    SelectBySingleFeaturePerformance,
    SelectByTargetMeanPerformance,
    SmartCorrelatedSelection,
    SelectByInformationValue,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from tests.estimator_checks.init_params_triggered_functionality_checks import (
    check_confirm_variables,
    check_raises_error_if_only_1_variable,
)

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
]

_multivariate_estimators = [
    DropDuplicateFeatures(),
    DropCorrelatedFeatures(),
    SmartCorrelatedSelection(),
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy", threshold=-100),
]

_univariate_estimators = [
    DropFeatures(features_to_drop=["var_1"]),
    DropConstantFeatures(missing_values="ignore"),
    DropHighPSIFeatures(bins=5),
    SelectByTargetMeanPerformance(bins=3, regression=False, threshold=0),
    SelectBySingleFeaturePerformance(
        estimator=_logreg, scoring="accuracy", threshold=0,
    ),
    SelectByInformationValue(bins=2),
]

_model_based_estimators = [
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy", threshold=-100),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


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
    # this selectors need an estimator as an input param
    # test error otherwise.
    with pytest.raises(TypeError):
        estimator()
