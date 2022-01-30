import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
)
from feature_engine.encoding import MeanEncoder
from feature_engine.prediction import TargetMeanClassifier

_false_input_params = [
    ("salsa", "arbitrary"),
    ("33", "mean-encoder"),
    ([7], True),
    (False, "prost"),
]

_not_a_df = [
    "not_a_df",
    [0, -1, -2, "tree"],
    pd.Series(["pen", "paper", 1984, "desk"], name="office_items"),
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

    transformer.fit(df_pred[["City", "Age"]], df_pred["Plays_Football"])

    # test init params
    assert transformer.variables is None
    assert transformer.bins == 7
    assert transformer.strategy == "equal_frequency"
    # test attributes
    assert transformer.variables_categorical_ == ["City"]
    assert transformer.variables_numerical_ == ["Age"]
    assert transformer.classes_ == [1, 0]
    assert type(transformer.pipeline_
                .named_steps["discretiser"]) == EqualFrequencyDiscretiser
    assert type(transformer.pipeline_
                .named_steps["encoder_num"]) == MeanEncoder
    assert type(transformer.pipeline_
                .named_steps["encoder_cat"]) == MeanEncoder
    assert transformer.n_features_in_ == 2


def test_classifier_prediction_results_with_all_numerical_variables(
        df_pred, df_pred_small
):
    pass


def test_classifier_score_calculation(df_pred, df_pred_small):
    pass