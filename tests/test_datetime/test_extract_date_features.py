from typing import Type
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.variable_manipulation import _convert_variables_to_datetime
from feature_engine.datetime import ExtractDateFeatures

def test_extract_date_features(df_vartypes2):
    original_columns = df_vartypes2.columns
    vars_dt     = ["dob", "doa"]
    vars_non_dt = ["Name", "Age"]
    vars_mix    = ["Name", "Age", "doa"]
    df_transformed_full = _convert_variables_to_datetime(df_vartypes2).join(
        pd.DataFrame({
            "dob_month":   [2,2,2,2],
            "dob_quarter": [1,1,1,1],
            "dob_semester":[1,1,1,1],
            "dob_year" :   [2020,2020,2020,2020],
            "dob_woty" :   [9,9,9,9],
            "doa_month":   [12,2,6,5],
            "doa_quarter": [4,1,2,2],
            "doa_semester":[2,1,1,1],
            "doa_year":    [2010,1945,2100,1999],
            "doa_woty":    [48,5,22,17]
        })
    )

    #check exceptions upon class instantiation
    with pytest.raises(ValueError):
        transformer = ExtractDateFeatures(features_to_extract="not_supported")
    with pytest.raises(ValueError):
        transformer = ExtractDateFeatures(variables = 3.519)

    #check transformer attributes
    transformer = ExtractDateFeatures()
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == None
    assert transformer.features_to_extract == ["year"]
    
    #check exceptions upon calling fit method
    with pytest.raises(ValueError):
        transformer.fit(df_vartypes2[vars_non_dt])
    with pytest.raises(TypeError):
        transformer.fit(df_vartypes2, variables = vars_mix)

    #check exceptions upon calling transform method
    transformer.fit(df_vartypes2)
    with pytest.raises(ValueError):
        transformer.transform(df_vartypes2[vars_dt])
    with pytest.raises(NotFittedError):
        ExtractDateFeatures().transform(df_vartypes2)

    #check default initialized transformer
    transformer = ExtractDateFeatures()
    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa"]
    assert transformer.n_features_in_ == 6

    pd.testing.assert_frame_equal(X, df_transformed_full[
        list(original_columns) + ["dob_year", "doa_year"]
    ])

    #check transformer with specified variables to process
    transformer = ExtractDateFeatures(variables = "doa")
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == "doa"
    assert transformer.features_to_extract == ["year"]

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["doa"]
    pd.testing.assert_frame_equal(
        X, df_transformed_full[list(original_columns) + ["doa_year"]]
    )

    #check transformer with specified date features to extract
    transformer = ExtractDateFeatures(features_to_extract=["semester", "week_of_the_year"])
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == None
    assert transformer.features_to_extract == ["semester", "week_of_the_year"]

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa"]
    pd.testing.assert_frame_equal(
        X, df_transformed_full[list(original_columns) + [
            "dob_semester", "doa_semester",
            "dob_woty", "doa_woty"]]
    )