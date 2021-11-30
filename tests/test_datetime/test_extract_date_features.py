import pandas as pd

from feature_engine.variable_manipulation import _convert_variables_to_datetime
from feature_engine.datetime import ExtractDateFeatures

def test_extract_date_features(df_vartypes2):
    original_columns = df_vartypes2.columns
    df_transformed_full = _convert_variables_to_datetime(df_vartypes2).join(
        pd.DataFrame({
            "dob_month":   [2,2,2,2],
            "dob_quarter": [1,1,1,1],
            "dob_semester":[1,1,1,1],
            "dob_year" :   [2020,2020,2020,2020],
            "doa_month":   [12,2,6,5],
            "doa_quarter": [4,1,2,2],
            "doa_semester":[2,1,1,1],
            "doa_year":    [2010,1945,2100,1999]
        })
    )

    transformer = ExtractDateFeatures()
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == None
    assert transformer.features_to_extract == ["year"]
    
    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa"]
    assert transformer.n_features_in_ == 6


    pd.testing.assert_frame_equal(X, df_transformed_full[
        list(original_columns) + ["dob_year", "doa_year"]
    ])

    transformer = ExtractDateFeatures(variables = "doa")
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == "doa"
    assert transformer.features_to_extract == ["year"]

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["doa"]
    pd.testing.assert_frame_equal(
        X, df_transformed_full[list(original_columns) + ["doa_year"]]
    )

    transformer = ExtractDateFeatures(features_to_extract="semester")
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == None

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa"]
    pd.testing.assert_frame_equal(
        X, df_transformed_full[list(original_columns) + [
            var+"_semester" for var in ("dob","doa")]]
    )