import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.datetime import ExtractDateFeatures


def test_extract_date_features(df_datetime):
    vars_dt = ["datetime_range", "date_obj1", "date_obj2"]
    vars_non_dt = ["Name", "Age"]
    vars_mix = ["Name", "Age", "date_obj1"]
    feat_names = [
        "month", "quarter", "semester", "year", "woty", "dotw",
        "dotm", "doty", "is_weekend", "wotm"
    ]
    df_transformed_full = df_datetime.join(
        pd.DataFrame(
            {
                "datetime_range_month": [2, 2, 2, 2],
                "datetime_range_quarter": [1, 1, 1, 1],
                "datetime_range_semester": [1, 1, 1, 1],
                "datetime_range_year": [2020, 2020, 2020, 2020],
                "datetime_range_woty": [9, 9, 9, 9],
                "datetime_range_dotw": [1, 2, 3, 4],
                "datetime_range_dotm": [24, 25, 26, 27],
                "datetime_range_doty": [55, 56, 57, 58],
                "datetime_range_is_weekend": [False, False, False, False],
                "datetime_range_wotm": [4, 4, 4, 4],
                "date_obj1_month": [12, 2, 6, 5],
                "date_obj1_quarter": [4, 1, 2, 2],
                "date_obj1_semester": [2, 1, 1, 1],
                "date_obj1_year": [2010, 1945, 2100, 1999],
                "date_obj1_woty": [48, 8, 24, 20],
                "date_obj1_dotw": [3, 6, 1, 1],
                "date_obj1_dotm": [1, 24, 14, 17],
                "date_obj1_doty": [335, 55, 165, 137],
                "date_obj1_is_weekend": [False, True, False, False],
                "date_obj1_wotm": [1, 4, 2, 3],
                "date_obj2_month": [10, 9, 5, 3],
                "date_obj2_quarter": [4, 3, 2, 1],
                "date_obj2_semester": [2, 2, 1, 1],
                "date_obj2_year": [2012, 2009, 1995, 2004],
                "date_obj2_woty": [41, 37, 21, 12],
                "date_obj2_dotw": [4, 3, 4, 3],
                "date_obj2_dotm": [11, 9, 25, 17],
                "date_obj2_doty": [285, 252, 145, 77],
                "date_obj2_is_weekend": [False, False, False, False],
                "date_obj2_wotm": [2, 2, 4, 3],
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
    assert ExtractDateFeatures(variables=["Age", "datetime_range"])\
        .variables == ["Age", "datetime_range"]
    transformer.fit(df_datetime)
    assert transformer.features_to_extract_ == transformer.supported
    transformer = ExtractDateFeatures(features_to_extract=["year"])
    transformer.fit(df_datetime)
    assert transformer.features_to_extract_ == ["year"]

    # check exceptions upon calling fit method
    transformer = ExtractDateFeatures()
    with pytest.raises(TypeError):
        transformer.fit("not_a_df")
    with pytest.raises(TypeError):
        ExtractDateFeatures(variables=["Age"]).fit(df_datetime)
    with pytest.raises(TypeError):
        ExtractDateFeatures(variables=vars_mix).fit(df_datetime)
    with pytest.raises(ValueError):
        transformer.fit(df_datetime[vars_non_dt])
    with pytest.raises(ValueError):
        transformer.fit(dates_nan)

    # check exceptions upon calling transform method
    transformer.fit(df_datetime)
    with pytest.raises(ValueError):
        transformer.transform(df_datetime[vars_dt])
    df_na = df_datetime.copy()
    df_na.loc[0, "date_obj1"] = np.nan
    with pytest.raises(ValueError):
        transformer.transform(df_na)
    with pytest.raises(NotFittedError):
        ExtractDateFeatures().transform(df_datetime)

    # check default initialized transformer
    transformer = ExtractDateFeatures()
    X = transformer.fit_transform(df_datetime)
    assert transformer.variables_ == ["datetime_range", "date_obj1", "date_obj2"]
    assert transformer.n_features_in_ == 5
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + [
            var + '_' + feat for var in vars_dt for feat in feat_names]]
    )

    # check transformer with specified variables to process
    transformer = ExtractDateFeatures(variables="date_obj1")
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == "date_obj1"
    assert transformer.features_to_extract is None

    X = transformer.fit_transform(df_datetime)
    assert transformer.variables_ == ["date_obj1"]
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + ["datetime_range", "date_obj2"] + [
            "date_obj1_" + feat for feat in feat_names]]
    )
    X = ExtractDateFeatures(variables=["datetime_range", "date_obj2"])\
        .fit_transform(df_datetime)
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + ["date_obj1"] + [
            var + "_" + feat
            for var in ["datetime_range", "date_obj2"]
            for feat in feat_names]]
    )
    X = ExtractDateFeatures(variables=["date_obj2", "date_obj1"])\
        .fit_transform(df_datetime)
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + ["datetime_range"] + [
            var + "_" + feat
            for var in ["date_obj2", "date_obj1"]
            for feat in feat_names]]
    )

    # check transformer with specified date features to extract
    transformer = ExtractDateFeatures(
        features_to_extract=["semester", "week_of_the_year"]
    )
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables is None
    assert transformer.features_to_extract == ["semester", "week_of_the_year"]

    X = transformer.fit_transform(df_datetime)
    assert transformer.variables_ == ["datetime_range", "date_obj1", "date_obj2"]
    pd.testing.assert_frame_equal(
        X,
        df_transformed_full[
            vars_non_dt
            + [
                "datetime_range_semester",
                "datetime_range_woty",
                "date_obj1_semester",
                "date_obj1_woty",
                "date_obj2_semester",
                "date_obj2_woty",
            ]
        ],
    )

    # check transformer with option to drop datetime features turned off
    X = ExtractDateFeatures(
        variables=["datetime_range", "date_obj2"],
        features_to_extract=["week_of_the_year", "quarter"],
        drop_datetime=False).fit_transform(df_datetime)

    pd.testing.assert_frame_equal(
        X,
        pd.concat(
            [df_transformed_full[column] for column in vars_non_dt]
            + [pd.to_datetime(df_datetime["datetime_range"]),
               df_datetime["date_obj1"],
               pd.to_datetime(df_datetime["date_obj2"])]
            + [df_transformed_full[feat] for feat in [
                var + "_" + feat
                for var in ["datetime_range", "date_obj2"]
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

    # check transformer with different pd.to_datetime options
    transformer = ExtractDateFeatures(
        features_to_extract=["day_of_the_month"],
        dayfirst=True
    )
    pd.testing.assert_frame_equal(
        transformer.fit_transform(df_datetime[["date_obj2"]]),
        pd.DataFrame({"date_obj2_dotm": [10, 9, 25, 17]})
    )

    transformer = ExtractDateFeatures(
        features_to_extract=["year"],
        yearfirst=True
    )
    pd.testing.assert_frame_equal(
        transformer.fit_transform(df_datetime[["date_obj2"]]),
        pd.DataFrame({"date_obj2_year": [2010, 2009, 1995, 2004]})
    )
