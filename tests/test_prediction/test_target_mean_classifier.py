import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.prediction import TargetMeanClassifier
from tests.test_prediction.conftest import df_pred

_false_input_params = [
    ("salsa", "arbitrary"),
    ("33", "mean-encoder"),
    ([7], True),
    (False, "prost"),
]

_not_a_df = [
    "not_a_df",
    [0, -1, -2, "tree"],
    df_pred["Studies"],
]


@pytest.mark.parametrize("_bins, _strategy", _false_input_params)
def test_raises_error_when_wrong_input_params(_bins, _strategy):
    with pytest.raises(ValueError):
        assert TargetMeanClassifier(bins=_bins)
    with pytest.raises(ValueError):
        assert TargetMeanClassifier(strategy=_strategy)


def test_default_params():
    transformer = TargetMeanClassifier()
    assert isinstance(transformer, TargetMeanClassifier)
    assert transformer.variables is None
    assert transformer.bins == 5
    assert transformer.strategy == "equal_width"