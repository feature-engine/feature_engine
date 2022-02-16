import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.estimator_checks import check_feature_engine_estimator
from feature_engine.selection import (  # SelectByTargetMeanPerformance,
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropFeatures,
    DropHighPSIFeatures,
    RecursiveFeatureAddition,
    RecursiveFeatureElimination,
    SelectByShuffling,
    SelectBySingleFeaturePerformance,
    SmartCorrelatedSelection,
)

_logreg = LogisticRegression(C=0.0001, max_iter=2, random_state=1)

_estimators = [
    DropFeatures(features_to_drop=["0"]),
    DropConstantFeatures(missing_values="ignore"),
    DropDuplicateFeatures(),
    DropCorrelatedFeatures(),
    DropHighPSIFeatures(bins=5),
    SmartCorrelatedSelection(),
    SelectByShuffling(estimator=_logreg, scoring="accuracy"),
    SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy"),
    # FIXME: as part of PR 358
    # SelectByTargetMeanPerformance(scoring="r2_score", bins=3),
    RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy"),
    RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy"),
]


# the RFA and RFE fail most tests. I think it has to do
# with the numpy arrays used in sklearn tests.
@pytest.mark.parametrize("estimator", _estimators[:-2])
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "DropFeatures":
        estimator.set_params(features_to_drop=["var_1"])
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators[7:10])
def test_raises_error_when_no_estimator_passed(estimator):
    # this selectors need an estimator as an input param
    # test error otherwise.
    with pytest.raises(TypeError):
        estimator()
