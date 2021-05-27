import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.discretisation import (
    ArbitraryDiscretiser,
    DecisionTreeDiscretiser,
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)


@pytest.mark.parametrize(
    "Estimator",
    [
        DecisionTreeDiscretiser(),
        EqualFrequencyDiscretiser(),
        EqualWidthDiscretiser(),
        ArbitraryDiscretiser(binning_dict={"0": [-np.Inf, 0, np.Inf]}),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
