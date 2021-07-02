import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.creation import (
    CombineWithReferenceFeature,
    CyclicalTransformer,
    MathematicalCombination,
)


@pytest.mark.parametrize(
    "Estimator",
    [
        MathematicalCombination(variables_to_combine=["0", "1"]),
        CombineWithReferenceFeature(
            variables_to_combine=["0"], reference_variables=["1"]
        ),
        CyclicalTransformer(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
