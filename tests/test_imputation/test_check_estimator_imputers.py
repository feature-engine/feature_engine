import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.imputation import (
    AddMissingIndicator,
    ArbitraryNumberImputer,
    CategoricalImputer,
    DropMissingData,
    EndTailImputer,
    MeanMedianImputer,
    RandomSampleImputer,
)


@pytest.mark.parametrize(
    "Estimator",
    [
        MeanMedianImputer(),
        ArbitraryNumberImputer(),
        CategoricalImputer(ignore_format=True),
        EndTailImputer(),
        AddMissingIndicator(),
        RandomSampleImputer(),
        DropMissingData(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
