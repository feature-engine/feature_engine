import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.discretisation import (
    ArbitraryDiscretiser,
    DecisionTreeDiscretiser,
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [
    DecisionTreeDiscretiser(regression=False),
    EqualFrequencyDiscretiser(),
    EqualWidthDiscretiser(),
    ArbitraryDiscretiser(binning_dict={"0": [-np.Inf, 0, np.Inf]}),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "ArbitraryDiscretiser":
        estimator.set_params(binning_dict={"var_1": [-np.Inf, 0, np.Inf]})
    return check_feature_engine_estimator(estimator)
