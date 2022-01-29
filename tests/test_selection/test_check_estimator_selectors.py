import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.estimator_checks import (
    check_all_types_variables_assignment,
    check_numerical_variables_assignment,
    check_raises_error_when_fitting_not_a_df,
    check_raises_error_when_transforming_not_a_df,
    check_raises_non_fitted_error,
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

_estimators = [
    DropFeatures(features_to_drop=["0"]),
    DropConstantFeatures(missing_values="ignore"),
    DropDuplicateFeatures(),
    DropCorrelatedFeatures(),
    DropHighPSIFeatures(bins=5),
    SmartCorrelatedSelection(),
    SelectByShuffling(
        LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
    ),
    SelectBySingleFeaturePerformance(
        LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
    ),
    RecursiveFeatureAddition(
        LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
    ),
    RecursiveFeatureElimination(
        LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
    ),
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
    SelectByShuffling(
        LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
    ),
    SelectBySingleFeaturePerformance(
        LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
    ),
    RecursiveFeatureAddition(
        LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
    ),
    RecursiveFeatureElimination(
        LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
    ),
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


@pytest.mark.parametrize("estimator", _estimators_for_all_vars)
def test_sel_tall_types_variables_assignment(estimator):
    check_all_types_variables_assignment(estimator)
