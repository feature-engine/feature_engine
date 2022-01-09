import pandas
import numpy

from feature_engine.prediction import TargetMeanPredictor


def test_target_mean_predictor_fit(df_pred):
    predictor = TargetMeanPredictor(
        variables=None,
        bins=5,
        strategy="equal-width"
    )

    predictor.fit(df_pred[["City", "Age"]], df_pred["Marks"])

    # test init params
    assert predictor.variables is None
    assert predictor.bins == 5
    assert predictor.strategy == "equal-width"
    # test fit params
    assert predictor.variables_ == ["City", "Age"]
    assert predictor.disc_mean_dict_ == {"Age": {0: 0.8, 1: 0.3, 2: 0.5, 3: 0.8, 4: 0.25}}

