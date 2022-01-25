import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
)
from feature_engine.encoding import MeanEncoder
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


def test_attributes_upon_fitting(df_pred):
    transformer = TargetMeanClassifier(
        variables=None,
        bins=7,
        strategy="equal_frequency"
    )

    transformer.fit(df_pred[["City", "Age"]], df_pred["Marks"])

    # test init params
    assert transformer.variables is None
    assert transformer.bins == 7
    assert transformer.strategy == "equal_frequency"
    # test attributes
    assert transformer.variables_categorical_ == ["City"]
    assert transformer.variables_numerical_ == ["Age"]
    assert transformer.classes_ == [1, 0]
    assert transformer.pipeline_ == Pipeline(steps=[
        ('discretiser', EqualFrequencyDiscretiser(
            q=7, return_object=True, variables=['Age'])),
        ('encoder_num', MeanEncoder(errors='raise', variables=['Age'])),
        ('encoder_cat', MeanEncoder(errors='raise', variables=['City']))])
    assert transformer.n_features_in_ == 2