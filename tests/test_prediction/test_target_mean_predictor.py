import pandas as pd

from feature_engine.prediction import TargetMeanPredictor


def test_target_mean_predictor_fit(df_pred):
    # Case 1: check all init params and class attributes
    predictor = TargetMeanPredictor(
        variables=None,
        bins=5,
        strategy="equal_width"
    )

    predictor.fit(df_pred[["City", "Age"]], df_pred["Marks"])

    # test init params
    assert predictor.variables is None
    assert predictor.bins == 5
    assert predictor.strategy == "equal-width"
    # test fit params
    assert predictor.variables_ == ["City", "Age"]
    assert predictor._pipeline["discretisation"].variables == ["Age"]
    assert predictor._pipeline["encoder_num"].encoder_dict_ == {
        "Age": {0: 0.8, 1: 0.3, 2: 0.5, 3: 0.8, 4: 0.25}
    }
    assert predictor._pipeline["encoder_cat"].encoder_dict_ == {
        "City": {"Bristol": 0.1,
                 "Liverpool": 0.5333333333333333,
                 "London": 0.6666666666666666,
                 "Manchester": 0.5333333333333333,
                 }
    }


def test_target_mean_predictor_transformation(df_pred, df_pred_small):
    # Case 2: Check transformation
    predictor = TargetMeanPredictor(
        variables=None,
        bins=5,
        strategy="equal_width"
    )

    predictor.fit(df_pred[["City", "Age"]], df_pred["Marks"])
    predictions = predictor.predict(df_pred_small[["City", "Age"]]).round(6)

    # test results
    assert (predictions == pd.Series(
        [0.483333, 0.583333, 0.391667, 0.666667, 0.3, 0.391667]
    )).all()


def test_regression_score_calculation_with_equal_distance(df_pred, df_pred_small):
    # Case 3: check score() method
    predictor = TargetMeanPredictor(
        variables=None,
        bins=5,
        strategy="equal_distance"
    )

    predictor.fit(df_pred[["City", "Age"]], df_pred["Marks"])
    r2 = predictor.score(
        df_pred_small[["City", "Age"]],
        df_pred_small["Marks"],
        regression=True
    )

    # test R-Squared calc
    assert r2.round(6) == -0.022365


def test_predictor_with_all_numerical_variables(df_pred, df_pred_small):
    # Case 4: Check predictor when all variables are numerical
    predictor = TargetMeanPredictorTest(
        variables=None,
        bins=3,
        strategy="equal_width"
    )

    predictor.fit(df_pred[["Age", "Height_cm"]], df_pred["Marks"])
    r2 = predictor.score(
        df_pred_small[["Age", "Height_cm"]], df_pred_small["Marks"], regression=True
    )

    # test R-Squared calc
    assert r2.round(6) == 0.132319