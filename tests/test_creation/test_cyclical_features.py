import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from numpy import array

from feature_engine.creation import CyclicalFeatures

CYCLICAL_DATA = {
    "day": [6, 7, 5, 3, 1, 2, 4],
    "months": [3, 7, 9, 12, 4, 6, 12],
}


def assert_df_equal(X, expected: dict) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert result[col] == pytest.approx(values, abs=1e-5)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_general_transformation_without_dropping_variables(make_df):
    # test case 1: just one variable.
    df = make_df(CYCLICAL_DATA)
    cyclical = CyclicalFeatures(variables=["day"])
    X = cyclical.fit_transform(df)

    expected = dict(CYCLICAL_DATA)
    expected["day_sin"] = [
        -0.78183,
        0.0,
        -0.97493,
        0.43388,
        0.78183,
        0.97493,
        -0.43388,
    ]
    expected["day_cos"] = [
        0.623490,
        1.0,
        -0.222521,
        -0.900969,
        0.623490,
        -0.222521,
        -0.900969,
    ]

    # fit attr
    assert cyclical.max_values_ == {"day": 7}

    # test transform output
    assert_df_equal(X, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_general_transformation_dropping_original_variables(make_df):
    # test case 1: just one variable, but dropping the variable after transformation
    df = make_df(CYCLICAL_DATA)
    cyclical = CyclicalFeatures(variables=["day"], drop_original=True)
    X = cyclical.fit_transform(df)

    expected = dict(CYCLICAL_DATA)
    expected["day_sin"] = [
        -0.78183,
        0.0,
        -0.97493,
        0.43388,
        0.78183,
        0.97493,
        -0.43388,
    ]
    expected["day_cos"] = [
        0.623490,
        1.0,
        -0.222521,
        -0.900969,
        0.623490,
        -0.222521,
        -0.900969,
    ]
    del expected["day"]

    # test fit attr
    assert cyclical.n_features_in_ == 2
    assert cyclical.max_values_ == {"day": 7}

    # test transform output
    assert_df_equal(X, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables(make_df):
    # test case 2: automatically select variables
    df = make_df(CYCLICAL_DATA)
    cyclical = CyclicalFeatures(variables=None, drop_original=True)
    X = cyclical.fit_transform(df)

    expected = {
        "day_sin": [
            -0.78183,
            0.0,
            -0.97493,
            0.43388,
            0.78183,
            0.97493,
            -0.43388,
        ],
        "day_cos": [
            0.62349,
            1.0,
            -0.222521,
            -0.900969,
            0.62349,
            -0.222521,
            -0.900969,
        ],
        "months_sin": [
            1.0,
            -0.5,
            -1.0,
            0.0,
            0.86603,
            0.0,
            0.0,
        ],
        "months_cos": [
            0.0,
            -0.86603,
            -0.0,
            1.0,
            -0.5,
            -1.0,
            1.0,
        ],
    }

    # test fit attr
    assert cyclical.max_values_ == {
        "day": 7,
        "months": 12,
    }

    # test transform output
    assert_df_equal(X, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_na_in_df(make_df):
    # test case 3: when dataset contains na, fit method
    df = make_df({"day": [1, 2, None, 4], "months": [1, 2, 3, 4]})
    msg = "Some of the variables in the dataset contain NaN"
    with pytest.raises(ValueError, match=msg):
        CyclicalFeatures().fit(df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_user_dictionary_key_not_in_df(make_df):
    df = make_df(CYCLICAL_DATA)
    # message differs by backend (pandas KeyError vs narwhals
    # ColumnNotFoundError, a KeyError subclass), so no match= here.
    with pytest.raises(KeyError):
        CyclicalFeatures(max_values={"dayi": 31}).fit(df)


def test_raises_error_when_init_parameters_not_permitted():
    msg = "The parameter can only take a dictionary or None"
    with pytest.raises(TypeError, match=msg):
        # when max_values is not a dictionary
        CyclicalFeatures(max_values=("dayi", 31))

    msg = "All values in the dictionary must be integer or float"
    with pytest.raises(ValueError, match=msg):
        # when max_values values are not integers or string
        CyclicalFeatures(max_values={"day": "31"})

    msg = "drop_original takes only boolean values True and False"
    with pytest.raises(ValueError, match=msg):
        # when drop original is not a boolean
        CyclicalFeatures(drop_original="True")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_max_values_mapping(make_df):
    df = make_df(CYCLICAL_DATA)
    cyclical = CyclicalFeatures(variables="day", max_values={"day": 31})

    X = cyclical.fit_transform(df)

    expected = dict(CYCLICAL_DATA)
    expected["day_sin"] = [
        0.937752,
        0.988468,
        0.848644,
        0.571268,
        0.201298,
        0.394355,
        0.724792,
    ]
    expected["day_cos"] = [
        0.347305,
        0.151428,
        0.528964,
        0.820763,
        0.979530,
        0.918958,
        0.688967,
    ]
    assert_df_equal(X, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "input_features", [None, ["day", "months"], array(["day", "months"])]
)
def test_get_feature_names_out(make_df, input_features):
    # default features from all variables
    df = make_df(CYCLICAL_DATA)
    transformer = CyclicalFeatures()
    X = transformer.fit_transform(df)
    feat_out = list(CYCLICAL_DATA.keys()) + [
        "day_sin",
        "day_cos",
        "months_sin",
        "months_cos",
    ]

    assert (
        list(nw.from_native(X, eager_only=True).columns)
        == transformer.get_feature_names_out()
    )
    assert transformer.get_feature_names_out(input_features=input_features) == feat_out

    msg = "input_features is not equal to feature_names_in_"
    with pytest.raises(ValueError, match=msg):
        transformer.get_feature_names_out(input_features=["day"])

    with pytest.raises(ValueError, match=msg):
        transformer.get_feature_names_out(input_features=["sandia", "banana"])

    transformer = CyclicalFeatures(drop_original=True)
    X = transformer.fit_transform(df)
    feat_out = ["day_sin", "day_cos", "months_sin", "months_cos"]
    assert (
        list(nw.from_native(X, eager_only=True).columns)
        == transformer.get_feature_names_out()
    )
    assert transformer.get_feature_names_out(input_features=input_features) == feat_out
