import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import check_estimator

from tests.estimator_checks.estimator_checks import (
    check_feature_engine_estimator,
)
from tests.estimator_checks.init_params_triggered_functionality_checks import (
    check_confirm_variables,
)
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
)

_logreg = LogisticRegression(C=0.0001, max_iter=2, random_state=1)

_estimators = [
    DropFeatures(features_to_drop=["0"]),
    DropConstantFeatures(missing_values="ignore"),
    DropDuplicateFeatures(),
    DropCorrelatedFeatures(),
    DropHighPSIFeatures(bins=5),
    SmartCorrelatedSelection(),
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectByTargetMeanPerformance(bins=3),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy"),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "DropFeatures":
        estimator.set_params(features_to_drop=["var_1"])
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_confirm_variables(estimator):
    if estimator.__class__.__name__ != "DropFeatures":
        return check_confirm_variables(estimator)


@pytest.mark.parametrize("estimator", _estimators[8:11])
def test_raises_error_when_no_estimator_passed(estimator):
    # this selectors need an estimator as an input param
    # test error otherwise.
    with pytest.raises(TypeError):
        estimator()
