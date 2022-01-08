import pandas
import numpy

from feature_engine.prediction import TargetMeanPredictor


def test_target_mean_predictor_fit(df_pred):
    predictor = TargetMeanPredictor(
        variables=["City", "Studies", "Age"],
        bins=5,
        strategy="equal-width"
    )
    predictor.fit(df_pred, df_pred["Marks"])

    # test init params
    assert predictor.variables == ["City", "Studies", "Age"]
    assert predictor.bins == 5
    assert predictor.strategy == "equal-width"

