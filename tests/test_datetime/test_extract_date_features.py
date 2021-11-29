from feature_engine.datetime import ExtractDateFeatures

def test_extract_date_features(df_vartypes2):

    transformer = ExtractDateFeatures()
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == None
    
    transformer.fit(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa"]

    X = transformer.transform(df_vartypes2)
    assert list(X.columns) == list(df_vartypes2.columns) + ["dob_month", "doa_month"]


    transformer = ExtractDateFeatures(variables = "doa")
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == "doa"

    #transformer.fit_transform(df_vartypes2)
    
    transformer.fit(df_vartypes2)
    assert transformer.variables_ == ["doa"]

    X = transformer.transform(df_vartypes2)
    assert list(X.columns) == list(df_vartypes2.columns) + ["doa_month"]