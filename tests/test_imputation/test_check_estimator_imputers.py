import pytest
from sklearn.utils.estimator_checks import check_estimator

from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from feature_engine.imputation import (
    AddMissingIndicator,
    ArbitraryNumberImputer,
    CategoricalImputer,
    DropMissingData,
    EndTailImputer,
    MeanMedianImputer,
    RandomSampleImputer,
)

_estimators = [
    MeanMedianImputer(),
    ArbitraryNumberImputer(),
    CategoricalImputer(fill_value=0, ignore_format=True),
    EndTailImputer(),
    AddMissingIndicator(),
    RandomSampleImputer(),
    DropMissingData(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "CategoricalImputer":
        estimator.set_params(ignore_format=False)
    if estimator.__class__.__name__ in ["DropMissingData", "AddMissingIndicator"]:
        estimator.set_params(missing_only=False)
    return check_feature_engine_estimator(estimator)
