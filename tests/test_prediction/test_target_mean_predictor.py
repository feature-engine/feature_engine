import pandas
import numpy

from feature_engine.prediction import TargetMeanPredictor


def test_target_mean_predictor_fit(df_enc_category_dtypes):
    predictor = TargetMeanPredictor()
    predictor.fit(df_enc_category_dtypes[["var_A", "var_B"]], df_enc_category_dtypes["target"])

    # test init params

