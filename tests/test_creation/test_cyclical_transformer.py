import pandas as pd
import pytest

from feature_engine.creation import CyclicalTransformer


@pytest.fixture
def df_cyclical():
    df = {
        "day": [
            6,
            7,
            5,
            3,
            1,
            2,
            4,
        ],
        "months": [
            3,
            7,
            9,
            12,
            4,
            6,
            12,
        ],
    }
    df = pd.DataFrame(df)

    return df


def test_general_transformation_without_dropping_variables(df_cyclical):
    # test case 1: just one variable.
    cyclical = CyclicalTransformer(variables=["day"])
    X = cyclical.fit_transform(df_cyclical)

    transf_df = df_cyclical.copy()

    # expected output
    transf_df["day_sin"] = [
        -0.78183,
        0.0,
        -0.97493,
        0.43388,
        0.78183,
        0.97493,
        -0.43388,
    ]
    transf_df["day_cos"] = [
        0.623490,
        1.0,
        -0.222521,
        -0.900969,
        0.623490,
        -0.222521,
        -0.900969,
    ]

    # test init params
    assert cyclical.variables == ["day"]

    # test fit attr
    assert cyclical.variables_ == ["day"]
    assert cyclical.n_features_in_ == 2
    assert cyclical.max_values_ == {
        "day": 7,
    }

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


def test_general_transformation_dropping_original_variables(df_cyclical):
    # test case 1: just one variable, but dropping the variable after transformation
    cyclical = CyclicalTransformer(variables=["day"], drop_original=True)
    X = cyclical.fit_transform(df_cyclical)

    transf_df = df_cyclical.copy()

    # expected output
    transf_df["day_sin"] = [
        -0.78183,
        0.0,
        -0.97493,
        0.43388,
        0.78183,
        0.97493,
        -0.43388,
    ]
    transf_df["day_cos"] = [
        0.623490,
        1.0,
        -0.222521,
        -0.900969,
        0.623490,
        -0.222521,
        -0.900969,
    ]
    transf_df = transf_df.drop(columns="day")

    # test init params
    assert cyclical.variables == ["day"]

    # test fit attr
    assert cyclical.variables_ == ["day"]
    assert cyclical.n_features_in_ == 2
    assert cyclical.max_values_ == {
        "day": 7,
    }

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


def test_automatically_find_variables(df_cyclical):
    # test case 2: automatically select variables
    cyclical = CyclicalTransformer(variables=None, drop_original=True)
    X = cyclical.fit_transform(df_cyclical)
    transf_df = df_cyclical.copy()

    # expected output
    transf_df["day_sin"] = [
        -0.78183,
        0.0,
        -0.97493,
        0.43388,
        0.78183,
        0.97493,
        -0.43388,
    ]
    transf_df["day_cos"] = [
        0.62349,
        1.0,
        -0.222521,
        -0.900969,
        0.62349,
        -0.222521,
        -0.900969,
    ]
    transf_df["months_sin"] = [
        1.0,
        -0.5,
        -1.0,
        0.0,
        0.86603,
        0.0,
        0.0,
    ]
    transf_df["months_cos"] = [
        0.0,
        -0.86603,
        -0.0,
        1.0,
        -0.5,
        -1.0,
        1.0,
    ]
    transf_df = transf_df.drop(columns=["day", "months"])

    # test init params
    assert cyclical.variables is None

    # test fit attr
    assert cyclical.variables_ == ["day", "months"]
    assert cyclical.n_features_in_ == 2
    assert cyclical.max_values_ == {
        "day": 7,
        "months": 12,
    }

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = CyclicalTransformer()
        transformer.fit(df_na)


def test_fit_raises_error_if_mapping_key_not_in_variables(df_cyclical):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = CyclicalTransformer(variables="day", max_values={"dayi": 31})
        transformer.fit(df_cyclical)


def test_check_validation_of_init_parameters(df_cyclical):

    with pytest.raises(TypeError):
        transformer = CyclicalTransformer(variables="day", max_values=("dayi", 31))
        transformer.fit(df_cyclical)

    with pytest.raises(TypeError):
        transformer = CyclicalTransformer(variables="day", max_values={"day": "31"})

    with pytest.raises(TypeError):
        transformer = CyclicalTransformer(variables="day", drop_original="True")


def test_max_values_mapping(df_cyclical):
    cyclical = CyclicalTransformer(variables="day", max_values={"day": 31})

    X = cyclical.fit_transform(df_cyclical)

    transf_df = df_cyclical.copy()
    transf_df["day_sin"] = [
        0.937752,
        0.988468,
        0.848644,
        0.571268,
        0.201298,
        0.394355,
        0.724792,
    ]
    transf_df["day_cos"] = [
        0.347305,
        0.151428,
        0.528964,
        0.820763,
        0.979530,
        0.918958,
        0.688967,
    ]
    pd.testing.assert_frame_equal(X, transf_df)


def test_get_feature_names_out(df_cyclical):
    # default features from all variables
    transformer = CyclicalTransformer()
    X = transformer.fit_transform(df_cyclical)
    assert list(X.columns) == transformer.get_feature_names_out()
    assert transformer.get_feature_names_out(input_features=["day"]) == [
        "day_sin",
        "day_cos",
    ]
    assert transformer.get_feature_names_out(input_features=["day", "months"]) == [
        "day_sin",
        "day_cos",
        "months_sin",
        "months_cos",
    ]

    # default features from 1 variable
    transformer = CyclicalTransformer(drop_original=True)
    X = transformer.fit_transform(df_cyclical)
    assert list(X.columns) == transformer.get_feature_names_out()
    assert transformer.get_feature_names_out(input_features=["day"]) == [
        "day_sin",
        "day_cos",
    ]
    assert transformer.get_feature_names_out(input_features=["day", "months"]) == [
        "day_sin",
        "day_cos",
        "months_sin",
        "months_cos",
    ]

    with pytest.raises(ValueError):
        # assert error when user passes a string instead of list
        transformer.get_feature_names_out(input_features="day")

    with pytest.raises(ValueError):
        # assert error when uses passes features that were not lagged
        transformer.get_feature_names_out(input_features=["color"])
