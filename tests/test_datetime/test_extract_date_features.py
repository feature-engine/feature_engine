import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.datetime import ExtractDateFeatures


def test_extract_date_features(df_vartypes2):
    vars_dt = ["dob", "doa", "dof"]
    vars_non_dt = ["Name", "City", "Age", "Marks"]
    vars_mix = ["Name", "Age", "doa"]
    feats_supported = [
            "month",
            "quarter",
            "semester",
            "year",
            "week_of_the_year",
            "day_of_the_week",
            "day_of_the_month",
            "is_weekend",
            "week_of_the_month",
        ]
    feat_names = [
        "month", "quarter", "semester", "year", "woty", "dotw",
        "dotm", "is_weekend", "wotm"
    ]
    df_transformed_full = df_vartypes2.join(
        pd.DataFrame(
            {
                "dob_month": [2, 2, 2, 2],
                "dob_quarter": [1, 1, 1, 1],
                "dob_semester": [1, 1, 1, 1],
                "dob_year": [2020, 2020, 2020, 2020],
                "dob_woty": [9, 9, 9, 9],
                "dob_dotw": [1, 2, 3, 4],
                "dob_dotm": [24, 25, 26, 27],
                "dob_is_weekend": [False, False, False, False],
                "dob_wotm": [4, 4, 4, 4],
                "doa_month": [12, 2, 6, 5],
                "doa_quarter": [4, 1, 2, 2],
                "doa_semester": [2, 1, 1, 1],
                "doa_year": [2010, 1945, 2100, 1999],
                "doa_woty": [48, 8, 24, 20],
                "doa_dotw": [3, 6, 1, 1],
                "doa_dotm": [1, 24, 14, 17],
                "doa_is_weekend": [False, True, False, False],
                "doa_wotm": [1, 4, 2, 3],
                "dof_month": [10, 9, 5, 3],
                "dof_quarter": [4, 3, 2, 1],
                "dof_semester": [2, 2, 1, 1],
                "dof_year": [2012, 2009, 1995, 2004],
                "dof_woty": [41, 37, 21, 12],
                "dof_dotw": [4, 3, 4, 3],
                "dof_dotm": [11, 9, 25, 17],
                "dof_is_weekend": [False, False, False, False],
                "dof_wotm": [2, 2, 4, 3],
            }
        )
    )
    dates_nan = pd.DataFrame({"dates_na": ["Feb-2010", np.nan, "Jun-1922", np.nan]})

    # check exceptions upon class instantiation
    with pytest.raises(ValueError):
        assert ExtractDateFeatures(features_to_extract=["not_supported"])
    with pytest.raises(ValueError):
        assert ExtractDateFeatures(features_to_extract=["year", 1874])
    with pytest.raises(ValueError):
        assert ExtractDateFeatures(variables=3.519)
    with pytest.raises(ValueError):
        assert ExtractDateFeatures(missing_values="wrong_option")
    with pytest.raises(TypeError):
        assert ExtractDateFeatures(features_to_extract="year")
    with pytest.raises(TypeError):
        assert ExtractDateFeatures(features_to_extract=14198)

    # check transformer attributes
    transformer = ExtractDateFeatures()
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables is None
    assert ExtractDateFeatures(variables="Age").variables == "Age"
    assert ExtractDateFeatures(variables=["Age", "dob"]).variables == ["Age", "dob"]
    transformer.fit(df_vartypes2)
    assert transformer.features_to_extract_ == transformer.supported
    transformer = ExtractDateFeatures(features_to_extract=["year"])
    transformer.fit(df_vartypes2)
    assert transformer.features_to_extract_ == ["year"]

    # check exceptions upon calling fit method
    transformer = ExtractDateFeatures()
    with pytest.raises(TypeError):
        transformer.fit("not_a_df")
    with pytest.raises(TypeError):
        ExtractDateFeatures(variables=["Age"]).fit(df_vartypes2)
    with pytest.raises(TypeError):
        ExtractDateFeatures(variables=vars_mix).fit(df_vartypes2)
    with pytest.raises(ValueError):
        transformer.fit(df_vartypes2[vars_non_dt])
    with pytest.raises(ValueError):
        transformer.fit(dates_nan)

    # check exceptions upon calling transform method
    transformer.fit(df_vartypes2)
    with pytest.raises(ValueError):
        transformer.transform(df_vartypes2[vars_dt])
    df_na = df_vartypes2.copy()
    df_na.loc[0, "doa"] = np.nan
    with pytest.raises(ValueError):
        transformer.transform(df_na)
    with pytest.raises(NotFittedError):
        ExtractDateFeatures().transform(df_vartypes2)

    # check default initialized transformer
    transformer = ExtractDateFeatures()
    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa", "dof"]
    assert transformer.n_features_in_ == 7
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + [
            var + '_' + feat for var in vars_dt for feat in feat_names]]
    )

    # check transformer with specified variables to process
    transformer = ExtractDateFeatures(variables="doa")
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == "doa"
    assert transformer.features_to_extract is None

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["doa"]
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + ["dob", "dof"] + [
            "doa_" + feat for feat in feat_names]]
    )
    X = ExtractDateFeatures(variables=["dob", "dof"]).fit_transform(df_vartypes2)
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + ["doa"] + [
            var + "_" + feat for var in ["dob", "dof"] for feat in feat_names]]
    )
    X = ExtractDateFeatures(variables=["dof", "doa"]).fit_transform(df_vartypes2)
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + ["dob"] + [
            var + "_" + feat for var in ["dof", "doa"] for feat in feat_names]]
    )

    # check transformer with specified date features to extract
    transformer = ExtractDateFeatures(
        features_to_extract=["semester", "week_of_the_year"]
    )
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables is None
    assert transformer.features_to_extract == ["semester", "week_of_the_year"]

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa", "dof"]
    pd.testing.assert_frame_equal(
        X,
        df_transformed_full[
            vars_non_dt
            + [
                "dob_semester",
                "dob_woty",
                "doa_semester",
                "doa_woty",
                "dof_semester",
                "dof_woty",
            ]
        ],
    )

    # check transformer with option to drop datetime features turned off
    X = ExtractDateFeatures(
        variables=["dob", "dof"],
        features_to_extract=["week_of_the_year", "quarter"],
        drop_datetime=False).fit_transform(df_vartypes2)

    pd.testing.assert_frame_equal(
        X,
        pd.concat(
            [df_transformed_full[column] for column in vars_non_dt]
            + [pd.to_datetime(df_vartypes2["dob"]),
               df_vartypes2["doa"],
               pd.to_datetime(df_vartypes2["dof"])]
            + [df_transformed_full[feat] for feat in [
                var + "_" + feat 
                for var in ["dob", "dof"] 
                for feat in ["quarter", "woty"]]],
            axis=1,
        ),
    )

    # check transformer with allowed nan option
    transformer = ExtractDateFeatures(
        features_to_extract=["year"],
        missing_values="ignore")
    pd.testing.assert_frame_equal(
        transformer.fit_transform(dates_nan),
        pd.DataFrame({"dates_na_year": [2010, np.nan, 1922, np.nan]})
    )