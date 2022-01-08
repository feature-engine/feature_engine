import pandas
import numpy

from feature_engine.prediction import TargetMeanPredictor


def test_target_mean_predictor_fit(df_pred):
    predictor = TargetMeanPredictor(
        variables=None,
        bins=5,
        strategy="equal-width"
    )
    print(df_pred[["City", "Studies", "Age"]].head())
    predictor.fit(df_pred[["City", "Studies", "Age"]], df_pred["Marks"])

    # test init params
    assert predictor.variables is None
    assert predictor.bins == 5
    assert predictor.strategy == "equal-width"

