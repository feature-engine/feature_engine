import pytest

from sklearn.utils.estimator_checks import check_estimator

from feature_engine.encoding import (
    CountFrequencyEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    WoEEncoder,
    PRatioEncoder,
)


# @pytest.mark.parametrize(
#     "Estimator", [
#         CountFrequencyEncoder(),
#         DecisionTreeEncoder(),
#         MeanEncoder(),
#         OneHotEncoder(),
#         OrdinalEncoder(),
#         RareLabelEncoder(),
#         WoEEncoder(),
#         PRatioEncoder(),
#     ]
# )
# def test_all_transformers(Estimator):
#     return check_estimator(Estimator)
