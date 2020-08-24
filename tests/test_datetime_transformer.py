
import pandas as pd
import pytest

from feature_engine.datetime_transformers import DateTimeTransformer


def test_datetime_default(dataframe_datetime_normal):

    transformer = DateTimeTransformer(variables=["var_A"])
    X = transformer.fit_transform(dataframe_datetime_normal)

    # expected
    df = pd.DataFrame({
        "var_B": ["John", "Jason", "Ethan"],
        "var_C": [25, 28, 30],
        "var_A_year": [2009, 2010, 2001],
        "var_A_month": [1, 8, 11],
        "var_A_day": [5, 9, 18],
        "var_A_quarter": [1, 3, 4],
        "var_A_semester": [1, 2, 2],
        "var_A_day_of_week": [0, 0, 6],
        "var_A_is_weekend": [0, 0, 1],
        "var_A_hr": [15, 0, 11],
        "var_A_min": [5, 0, 15],
        "var_A_sec": [2, 0, 0],

    })

    # init parameters
    assert transformer.variables == ["var_A"]
    # transform
    assert transformer.features_to_add == {
        "var_A": list(transformer.feature_map.keys())
    }
    assert transformer.keep_original is False
    # ignore the order of columns?
    pd.testing.assert_frame_equal(X, df, check_like=True)


def test_datetime_select_list(dataframe_datetime_normal):
    _features_to_add = ["year", "month", "day"]
    transformer = DateTimeTransformer(
        variables=["var_A"],
        features_to_add=_features_to_add,
        keep_original=True,
    )
    X = transformer.fit_transform(dataframe_datetime_normal)

    # expected
    df = pd.DataFrame({
        "var_A": ["2009-1-5 15:05:02", "2010-08-09", "2001-11-18 11:15:00", ],
        "var_B": ["John", "Jason", "Ethan"],
        "var_C": [25, 28, 30],
        "var_A_year": [2009, 2010, 2001],
        "var_A_month": [1, 8, 11],
        "var_A_day": [5, 9, 18],
    })

    # init parameters
    assert transformer.variables == ["var_A"]
    # transform
    assert transformer.features_to_add == {
        "var_A": _features_to_add
    }
    assert transformer.keep_original is True
    pd.testing.assert_frame_equal(X, df, check_like=True)


def test_datetime_select_dict_default(dataframe_datetime_multiple):
    transformer = DateTimeTransformer(
        variables=["var_A", "var_D"],
    )
    X = transformer.fit_transform(dataframe_datetime_multiple)

    # expected
    df = pd.DataFrame({
        "var_A_year": [2009, 2010, 2001],
        "var_A_month": [1, 8, 11],
        "var_A_day": [5, 9, 18],
        "var_A_quarter": [1, 3, 4],
        "var_A_semester": [1, 2, 2],
        "var_A_day_of_week": [0, 0, 6],
        "var_A_is_weekend": [0, 0, 1],
        "var_A_hr": [15, 0, 11],
        "var_A_min": [5, 0, 15],
        "var_A_sec": [2, 0, 0],
        "var_B": ["John", "Jason", "Ethan"],
        "var_C": [25, 28, 30],
        "var_D_year": [2011, 2000, 2009],
        "var_D_month": [6, 11, 8],
        "var_D_day": [5, 25, 15],
        "var_D_quarter": [2, 4, 3],
        "var_D_semester": [1, 2, 2],
        "var_D_day_of_week": [6, 5, 5],
        "var_D_is_weekend": [1, 1, 1],
        "var_D_hr": [6, 12, 13],
        "var_D_min": [5, 5, 1],
        "var_D_sec": [59, 21, 36],

    })

    # init parameters
    assert transformer.variables == ["var_A", "var_D"]
    # transform
    assert transformer.features_to_add == {
        "var_A": list(transformer.feature_map.keys()),
        "var_D": list(transformer.feature_map.keys())
    }
    assert transformer.keep_original is False
    pd.testing.assert_frame_equal(X, df, check_like=True)


def test_datetime_select_dict_specific(dataframe_datetime_multiple):
    transformer = DateTimeTransformer(
        variables=["var_A", "var_D"],
        features_to_add={
            "var_A": ["year", "month", "day"],
            "var_D": ["is_weekend", "semester", "hr", "min", "sec"]
        }
    )
    X = transformer.fit_transform(dataframe_datetime_multiple)

    # expected
    df = pd.DataFrame({
        "var_A_year": [2009, 2010, 2001],
        "var_A_month": [1, 8, 11],
        "var_A_day": [5, 9, 18],
        "var_B": ["John", "Jason", "Ethan"],
        "var_C": [25, 28, 30],
        "var_D_semester": [1, 2, 2],
        "var_D_is_weekend": [1, 1, 1],
        "var_D_hr": [6, 12, 13],
        "var_D_min": [5, 5, 1],
        "var_D_sec": [59, 21, 36],

    })

    # init parameters
    assert transformer.variables == ["var_A", "var_D"]
    # transform
    assert transformer.features_to_add == {
        "var_A": ["year", "month", "day"],
        "var_D": ["is_weekend", "semester", "hr", "min", "sec"]
    }
    assert transformer.keep_original is False
    pd.testing.assert_frame_equal(X, df, check_like=True)


def test_datetime_select_non_existent_columns(dataframe_datetime_multiple):
    with pytest.raises(KeyError):
        transformer = DateTimeTransformer(
            variables=["var_X"]
        )
        X = transformer.fit_transform(dataframe_datetime_multiple)

    with pytest.raises(ValueError):
        transformer = DateTimeTransformer(
            variables=["var_A"],
            features_to_add={
                "var_X": ["year"]
            }
        )
        X = transformer.fit_transform(dataframe_datetime_multiple)


def test_datetime_non_existent_features(dataframe_datetime_normal):
    with pytest.raises(ValueError):
        transformer = DateTimeTransformer(
            variables=["var_A"],
            features_to_add=["year", "month", "nanosecond"]
        )
        X = transformer.fit_transform(dataframe_datetime_normal)

    with pytest.raises(ValueError):
        transformer = DateTimeTransformer(
            variables=["var_A"],
            features_to_add={
                "var_A": ["year", "month", "nanosecond"]
        })
        X = transformer.fit_transform(dataframe_datetime_normal)


def test_datetime_invalid_error(dataframe_datetime_invalid):
    with pytest.raises(ValueError):
        transformer = DateTimeTransformer(
            variables=["var_A"],
            features_to_add=["year", "month", "day"],
        )
        X = transformer.fit_transform(dataframe_datetime_invalid)


