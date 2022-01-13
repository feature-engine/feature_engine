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
    assert predictor._pipeline["discretisation"].variables == ["Age"]
    assert predictor._pipeline["encoder_num"].encoder_dict_ == {"Age": {0: 0.8, 1: 0.3, 2: 0.5, 3: 0.8, 4: 0.25}}
    assert predictor._pipeline["encoder_cat"].encoder_dict_ == {
        "City": {"Bristol": 0.1,
                 "Liverpool": 0.5333333333333333,
                 "London": 0.6666666666666666,
                 "Manchester": 0.5333333333333333,
                 }
    }
