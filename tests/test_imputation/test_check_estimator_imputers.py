import pytest

from sklearn.utils.estimator_checks import check_estimator

from feature_engine.imputation import (
    MeanMedianImputer,
    ArbitraryNumberImputer,
    CategoricalImputer,
    EndTailImputer,
    AddMissingIndicator,
    RandomSampleImputer,
    DropMissingData,
)


@pytest.mark.parametrize(
    "Estimator", [
        MeanMedianImputer(),
        ArbitraryNumberImputer(),
        # CategoricalImputer(),
        EndTailImputer(),
        AddMissingIndicator(),
        RandomSampleImputer(),
        DropMissingData(),
    ]
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)