import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.creation import (
    CyclicalFeatures,
    MathFeatures,
    RelativeFeatures,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [
    MathFeatures(variables=["x0", "x1"], func="mean", missing_values="ignore"),
    RelativeFeatures(
        variables=["x0", "x1"], reference=["x0"], func=["add"], missing_values="ignore"
    ),
    CyclicalFeatures(),
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
