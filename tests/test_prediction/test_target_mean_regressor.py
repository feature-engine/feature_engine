import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from feature_engine.discretisation import (
    EqualWidthDiscretiser,
)
from feature_engine.encoding import MeanEncoder
from feature_engine.prediction import TargetMeanRegressor
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
    pd.Series(["pen", "paper", 1984, "desk"], name="office_items"),
]


@pytest.mark.parametrize("_bins, _strategy", _false_input_params)
def test_raises_error_when_wrong_input_params(_bins, _strategy):
    with pytest.raises(ValueError):
        assert TargetMeanRegressor(bins=_bins)
    with pytest.raises(ValueError):
        assert TargetMeanRegressor(strategy=_strategy)


def test_default_params():
    transformer = TargetMeanRegressor()
    assert isinstance(transformer, TargetMeanRegressor)
    assert transformer.variables is None
    assert transformer.bins == 5
    assert transformer.strategy == "equal_width"


def test_attributes_upon_fitting(df_pred):
    # Case 1: check all init params and class attributes
    transformer = TargetMeanRegressor(
        variables=None,
        bins=5,
        strategy="equal_width"
    )
    transformer.fit(df_pred[["City", "Age", "Studies"]], df_pred["Marks"])

    # test init params
    assert transformer.variables is None
    assert transformer.bins == 5
    assert transformer.strategy == "equal_width"
    # test attributes
    assert transformer.variables_categorical_ == ["City", "Studies"]
    assert transformer.variables_numerical_ == ["Age"]
    assert transformer.pipeline_.named_steps == {
        'discretiser': EqualWidthDiscretiser(bins=5, return_object=True, variables=['Age']),
        'encoder_num': MeanEncoder(errors='raise', variables=['Age']),
        'encoder_cat': MeanEncoder(errors='raise', variables=['City', 'Studies']),
    }
    assert transformer.n_features_in_ == 3


def test_target_mean_predictor_transformation(df_pred, df_pred_small):
    # Case 2: Check transformation
    transformer = TargetMeanRegressor(
        variables=None,
        bins=5,
        strategy="equal_width"
    )

    transformer.fit(df_pred[["City", "Age"]], df_pred["Marks"])
    predictions = transformer.predict(df_pred_small[["City", "Age"]]).round(6)

    # test results
    assert (predictions == pd.Series(
        [0.483333, 0.583333, 0.391667, 0.666667, 0.3, 0.391667]
    )).all()


def test_regression_score_calculation_with_equal_frequency(df_pred, df_pred_small):
    # Case 3: check score() method
    transformer = TargetMeanRegressor(
        variables=None,
        bins=5,
        strategy="equal_frequency"
    )

    transformer.fit(df_pred[["City", "Age"]], df_pred["Marks"])
    r2 = transformer.score(
        df_pred_small[["City", "Age"]],
        df_pred_small["Marks"]
    )

    # test R-Squared calc
    assert r2.round(6) == -0.022365


def test_predictor_with_all_numerical_variables(df_pred, df_pred_small):
    # Case 4: Check predictor when all variables are numerical
    transformer = TargetMeanRegressor(
        variables=None,
        bins=3,
        strategy="equal_width"
    )

    transformer.fit(df_pred[["Age", "Height_cm"]], df_pred["Marks"])
    r2 = transformer.score(
        df_pred_small[["Age", "Height_cm"]], df_pred_small["Marks"]
    )

    # test R-Squared calc
    assert r2.round(6) == 0.132319


def test_non_fitted_error(df_pred):
    # case 6: test if transformer has been fitted
    with pytest.raises(NotFittedError):
        TargetMeanRegressor().predict(df_pred[["Studies", "Age"]])


def test_raises_error_when_df_has_nan(df_enc_na):
    # case 9: when dataset contains na, fit method
    with pytest.raises(ValueError):
        TargetMeanRegressor().fit(
            df_enc_na[["var_A", "var_B"]], df_enc_na["target"]
        )


def test_error_if_df_contains_na_in_transform(df_enc, df_enc_na):
    # case 10: when dataset contains na, transform method
    random = np.random.RandomState(42)
    y = random.normal(0, 3, len(df_enc))

    transformer = TargetMeanRegressor()
    transformer.fit(df_enc[["var_A", "var_B"]], y)
    with pytest.raises(ValueError):
        transformer.predict(df_enc_na[["var_A", "var_B"]])


def test_predictor_with_one_numerical_variable(df_pred, df_pred_small):
    # case 11: class properly executes w/ one numerical variable
    """
    transformer = TargetMeanRegressor()
    transformer.fit(df_pred["Age"], df_pred["Height_cm"])
    r2 = transformer.score(
        df_pred_small["Age"], df_pred_small["Height_cm"], regression=True
    )
    """


@pytest.mark.parametrize("_not_a_df", _not_a_df)
def test_raises_error_when_not_fitting_a_df(_not_a_df, df_pred):
    transformer = TargetMeanRegressor()
    with pytest.raises(TypeError):
        transformer.fit(_not_a_df, df_pred["Marks"])


@pytest.mark.parametrize("_not_a_df", _not_a_df)
def test_raises_error_when_not_transforming_a_df(_not_a_df, df_pred):
    transformer = TargetMeanRegressor()
    transformer.fit(df_pred[["Studies", "Age"]], df_pred["Marks"])
    with pytest.raises(TypeError):
        transformer.predict(_not_a_df)