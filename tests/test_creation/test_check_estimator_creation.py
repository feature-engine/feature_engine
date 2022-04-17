import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.creation import (
    CyclicalFeatures,
    MathFeatures,
    RelativeFeatures,
    # FIXME: remove in version 1.4
    CombineWithReferenceFeature,
    CyclicalTransformer,
    MathematicalCombination,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [
    MathFeatures(variables=["0", "1"], func="mean", missing_values="ignore"),
    RelativeFeatures(
        variables=["0", "1"], reference=["0"], func=["add"], missing_values="ignore"
    ),
    CyclicalFeatures(),
    # FIXME: remove in version 1.4
    MathematicalCombination(variables_to_combine=["0", "1"]),
    CombineWithReferenceFeature(variables_to_combine=["0"], reference_variables=["1"]),
    CyclicalTransformer(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


_estimators = [
    MathFeatures(variables=["var_1", "var_2", "var_3"], func="mean"),
    RelativeFeatures(variables=["var_1", "var_2"], reference=["var_3"], func=["add"]),
    CyclicalFeatures(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)
