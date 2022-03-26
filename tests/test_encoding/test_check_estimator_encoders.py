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
from feature_engine.estimator_checks import check_feature_engine_estimator

_estimators = [
    CountFrequencyEncoder(ignore_format=True),
    DecisionTreeEncoder(regression=False, ignore_format=True),
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
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


_estimators = [
    CountFrequencyEncoder(),
    DecisionTreeEncoder(regression=False),
    MeanEncoder(),
    OneHotEncoder(),
    OrdinalEncoder(),
    RareLabelEncoder(),
    WoEEncoder(),
    PRatioEncoder(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)
