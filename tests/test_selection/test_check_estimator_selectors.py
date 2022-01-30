import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.estimator_checks import (
    check_all_types_variables_assignment,
    check_error_if_y_not_passed,
    check_error_param_missing_values,
    check_numerical_variables_assignment,
    check_raises_error_when_fitting_not_a_df,
    check_raises_error_when_transforming_not_a_df,
    check_raises_non_fitted_error,
    check_takes_cv_constructor,
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

_logreg = LogisticRegression(max_iter=2, random_state=1)

_estimators = [
    DropFeatures(features_to_drop=["0"]),
    DropConstantFeatures(missing_values="ignore"),
    DropDuplicateFeatures(),
    DropCorrelatedFeatures(),
    DropHighPSIFeatures(bins=5),
    SmartCorrelatedSelection(),
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy"),
    SelectByTargetMeanPerformance(scoring="r2_score", bins=3),
]


@pytest.mark.parametrize("Estimator", _estimators)
def test_check_estimator_from_sklearn(Estimator):
    return check_estimator(Estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_sel_raises_non_fitted_error(estimator):
    check_raises_non_fitted_error(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_sel_raises_error_when_fitting_not_a_df(estimator):
    check_raises_error_when_fitting_not_a_df(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_sel_raises_error_when_transforming_not_a_df(estimator):
    if estimator.__class__.__name__ == "DropFeatures":
        estimator.set_params(features_to_drop=["var_1"])
    check_raises_error_when_transforming_not_a_df(estimator)


_estimators_for_numerical_vars = [
    DropCorrelatedFeatures(),
    DropHighPSIFeatures(bins=5),
    SmartCorrelatedSelection(),
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy"),
]


@pytest.mark.parametrize("estimator", _estimators_for_numerical_vars)
def test_sel_numerical_variables_assignment(estimator):
    check_numerical_variables_assignment(estimator)


_estimators_for_all_vars = [
    DropConstantFeatures(),
    DropDuplicateFeatures(),
    # TODO: below test is not passing, something is wrong
    # SelectByTargetMeanPerformance(),
]


_estimators_require_y = [
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy"),
    SelectByTargetMeanPerformance(scoring="r2_score", bins=3),
]


@pytest.mark.parametrize("estimator", _estimators_require_y)
def test_error_if_y_not_passed(estimator):
    check_error_if_y_not_passed(estimator)


@pytest.mark.parametrize("estimator", _estimators_for_all_vars)
def test_sel_tall_types_variables_assignment(estimator):
    check_all_types_variables_assignment(estimator)


_estimators_with_cv = [
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    # TODO: test is not passing
    # RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy"),
    SmartCorrelatedSelection(estimator=_logreg),
]


@pytest.mark.parametrize("estimator", _estimators_with_cv)
def test_takes_cv_constructor(estimator):
    check_takes_cv_constructor(estimator)


_estimators_with_missing_allowed = [
    DropConstantFeatures,
    DropDuplicateFeatures,
    DropCorrelatedFeatures,
    DropHighPSIFeatures,
    SmartCorrelatedSelection,
]


@pytest.mark.parametrize("estimator", _estimators_with_missing_allowed)
def test_error_param_missing_values(estimator):
    check_error_param_missing_values(estimator)
