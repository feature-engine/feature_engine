import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.selection import (  # SelectByTargetMeanPerformance,
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropFeatures,
    RecursiveFeatureAddition,
    RecursiveFeatureElimination,
    SelectByShuffling,
    SelectBySingleFeaturePerformance,
    SmartCorrelatedSelection,
)


@pytest.mark.parametrize(
    "Estimator",
    [
        DropFeatures(features_to_drop=["0"]),
        DropConstantFeatures(),
        DropDuplicateFeatures(),
        DropCorrelatedFeatures(),
        SmartCorrelatedSelection(),
        SelectByShuffling(RandomForestClassifier(random_state=1), scoring="accuracy"),
        SelectBySingleFeaturePerformance(
            RandomForestClassifier(random_state=1), scoring="accuracy"
        ),
        RecursiveFeatureAddition(
            RandomForestClassifier(random_state=1), scoring="accuracy"
        ),
        RecursiveFeatureElimination(
            RandomForestClassifier(random_state=1), scoring="accuracy"
        ),
        # SelectByTargetMeanPerformance(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
