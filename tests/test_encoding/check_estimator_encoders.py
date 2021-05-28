import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.encoding import (
    CountFrequencyEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PRatioEncoder,
    RareLabelEncoder,
    WoEEncoder,
)


@pytest.mark.parametrize(
    "Estimator",
    [
        CountFrequencyEncoder(ignore_format=True),
        DecisionTreeEncoder(ignore_format=True),
        MeanEncoder(ignore_format=True),
        OneHotEncoder(ignore_format=True),
        OrdinalEncoder(ignore_format=True),
        RareLabelEncoder(
            tol=0.00000000001,
            n_categories=100000000000,
            replace_with=10,
            ignore_format=True,
        ),
        WoEEncoder(ignore_format=True),
        PRatioEncoder(ignore_format=True),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
