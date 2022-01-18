import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.prediction import TargetMeanRegressor


def test_target_mean_predictor_fit(df_pred):
    # Case 1: check all init params and class attributes
    transformer = TargetMeanRegressor(
        variables=None,
        bins=5,
        strategy="equal_width"
    )

    transformer.fit(df_pred[["City", "Age"]], df_pred["Marks"])

    # test init params
    assert transformer.variables is None
    assert transformer.bins == 5
    assert transformer.strategy == "equal_width"
    # test fit params
    assert transformer.variables_ == ["City", "Age"]
    assert transformer._pipeline["discretisation"].variables == ["Age"]
    assert transformer._pipeline["encoder_num"].encoder_dict_ == {
        "Age": {0: 0.8, 1: 0.3, 2: 0.5, 3: 0.8, 4: 0.25}
    }
    assert transformer._pipeline["encoder_cat"].encoder_dict_ == {
        "City": {"Bristol": 0.1,
                 "Liverpool": 0.5333333333333333,
                 "London": 0.6666666666666666,
                 "Manchester": 0.5333333333333333,
                 }
    }


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


def test_regression_score_calculation_with_equal_distance(df_pred, df_pred_small):
    # Case 3: check score() method
    transformer = TargetMeanRegressor(
        variables=None,
        bins=5,
        strategy="equal_distance"
    )

    transformer.fit(df_pred[["City", "Age"]], df_pred["Marks"])
    r2 = transformer.score(
        df_pred_small[["City", "Age"]],
        df_pred_small["Marks"],
        regression=True
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
        df_pred_small[["Age", "Height_cm"]], df_pred_small["Marks"], regression=True
    )

    # test R-Squared calc
    assert r2.round(6) == 0.132319


def test_predictor_with_all_categorical_variables(df_pred, df_pred_small):
    # Case 5: Check predictor when all variables are categorical
    """
    transformer = TargetMeanRegressorTest(
        variables=None,
        bins=3,
        strategy="equal_width"
    )

    transformer.fit(df_pred[["City", "Studies"]], df_pred["Plays_Football"])
    accuracy_score = transformer.score(
        df_pred_small[["City", "Studies"]],
        df_pred_small["Plays_Football"],
        regression=False
    )

    # NEED TO UPDATE VALUE
    assert accuracy_score.round(6) == 0
    """
    pass


def test_non_fitted_error(df_pred):
    # case 6: test if transformer has been fitted
    with pytest.raises(NotFittedError):
        transformer = TargetMeanRegressor()
        transformer.predict(df_pred[["Studies", "Age"]])


def test_incorrect_strategy_during_instantiation(df_pred):
    # case 7: test if inappropriate value has been inputted for the
    # 'strategy' param
    with pytest.raises(ValueError):
        transformer = TargetMeanRegressor(strategy="arbitrary")
        transformer.fit(df_pred[["City", "Studies"]], df_pred["Marks"])


def test_incorrect_bin_value_during_instantiation(df_pred):
    # case 8: test if an inappropriate value has been inputted for the
    # 'bins' param
    msg = "Got lamp bins instead of an integer."
    with pytest.raises(TypeError) as record:
        transformer = TargetMeanRegressor(bins="lamp")
        transformer.fit(df_pred[["City", "Studies"]], df_pred["Marks"])

    assert str(record.value) == msg


def test_error_if_df_contains_na_in_fit(df_enc_na):
    # case 9: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = TargetMeanRegressor()
        transformer.fit(df_enc_na[["var_A", "var_B"]], df_enc_na["target"])


def test_error_if_df_contains_na_in_transform(df_enc, df_enc_na):
    # case 10: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = TargetMeanRegressor()
        transformer.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        y_pred = transformer.predict(df_enc_na[["var_A", "var_B"]])
        return y_pred


def test_predictor_with_one_numerical_variable(df_pred, df_pred_small):
    # case 11: class properly executes w/ one numerical variable
    """
    transformer = TargetMeanRegressor()
    transformer.fit(df_pred["Age"], df_pred["Height_cm"])
    r2 = transformer.score(
        df_pred_small["Age"], df_pred_small["Height_cm"], regression=True
    )
    """


def test_error_when_x_is_not_a_dataframe(df_pred):
    # case 12: return error if 'X' is not a dataframe
    msg = "X is not a pandas dataframe. The dataset should be a pandas dataframe."
    with pytest.raises(TypeError) as record:
        transformer = TargetMeanRegressor()
        transformer.fit(df_pred["Studies"], df_pred["Marks"])

    # check that error message matches
    assert str(record.value) == msg
