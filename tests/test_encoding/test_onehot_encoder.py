import re

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.encoding import OneHotEncoder
from tests.backend_helpers import frame_to_dict

DATA_ENC_BINARY = {
    "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "var_C": ["AHA"] * 12 + ["UHU"] * 8,
    "var_D": ["OHO"] * 5 + ["EHE"] * 15,
    "var_num": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}


# init parameters
@pytest.mark.parametrize("top_cat", ["empanada", [1], 0.5, -1])
def test_error_if_top_categories_not_integer(top_cat):
    msg = f"top_categories takes only positive integers. Got {top_cat} instead"
    with pytest.raises(ValueError, match=re.escape(msg)):
        OneHotEncoder(top_categories=top_cat)


@pytest.mark.parametrize("drop_last", ["empanada", [1], 0.5, -1, 1, None])
def test_error_if_drop_last_not_bool(drop_last):
    msg = f"drop_last takes only True or False. Got {drop_last} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        OneHotEncoder(drop_last=drop_last)


@pytest.mark.parametrize("drop_binary", ["hello", ["auto"], -1, 100, 0.5, None])
def test_error_if_drop_last_binary_not_bool(drop_binary):
    msg = f"drop_last_binary takes only True or False. Got {drop_binary} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        OneHotEncoder(drop_last_binary=drop_binary)


@pytest.mark.parametrize(
    "top_categories, drop_last, drop_last_binary, ignore_format",
    [
        (None, False, False, False),
        (1, True, False, True),
        (10, False, True, False),
        (0, True, True, True),
    ],
)
def test_init_param_assignment(
    top_categories, drop_last, drop_last_binary, ignore_format
):
    encoder = OneHotEncoder(
        top_categories=top_categories,
        drop_last=drop_last,
        drop_last_binary=drop_last_binary,
        ignore_format=ignore_format,
    )
    assert encoder.top_categories == top_categories
    assert encoder.drop_last is drop_last
    assert encoder.drop_last_binary is drop_last_binary
    assert encoder.ignore_format is ignore_format


# fit and transform
@pytest.mark.parametrize("index_", [[1, 2, 3], [3, 2, 1], [4, 9, 2]])
def test_concat_with_non_ordered_index(make_df, index_):
    data = {"varA": ["a", "b", "c"], "varB": ["d", "d", "a"]}
    # only pandas has a row index to scramble
    if make_df is pd.DataFrame:
        df = make_df(data, index=index_)
    else:
        df = make_df(data)
    encoder = OneHotEncoder()
    dft = encoder.fit_transform(df)

    expected = {
        "varA_a": [1, 0, 0],
        "varA_b": [0, 1, 0],
        "varA_c": [0, 0, 1],
        "varB_d": [1, 1, 0],
        "varB_a": [0, 0, 1],
    }
    assert isinstance(dft, make_df)
    assert list(dft.columns) == list(expected)
    assert frame_to_dict(dft) == expected


def test_encode_categories_in_k_binary_plus_select_vars_automatically(
    make_df, data_enc_big
):
    # test case 1: encode all categories into k binary variables, select variables
    # automatically
    encoder = OneHotEncoder(top_categories=None, variables=None, drop_last=False)
    X = encoder.fit_transform(make_df(data_enc_big))

    # test fit attr
    transf = {
        "var_A_A": 6, "var_A_B": 10, "var_A_C": 4, "var_A_D": 10, "var_A_E": 2,
        "var_A_F": 2, "var_A_G": 6, "var_B_A": 10, "var_B_B": 6, "var_B_C": 4,
        "var_B_D": 10, "var_B_E": 2, "var_B_F": 2, "var_B_G": 6, "var_C_A": 4,
        "var_C_B": 6, "var_C_C": 10, "var_C_D": 10, "var_C_E": 2, "var_C_F": 2,
        "var_C_G": 6,
    }

    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.variables_binary_ == []
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == {
        "var_A": ["A", "B", "C", "D", "E", "F", "G"],
        "var_B": ["A", "B", "C", "D", "E", "F", "G"],
        "var_C": ["A", "B", "C", "D", "E", "F", "G"],
    }
    # test transform output
    assert isinstance(X, make_df)
    result = frame_to_dict(X)
    assert {col: sum(result[col]) for col in transf} == transf
    assert "var_A" not in result


def test_encode_categories_in_k_minus_1_binary_plus_list_of_variables(
    make_df, data_enc_big
):
    # test case 2: encode all categories into k-1 binary variables,
    # pass list of variables
    encoder = OneHotEncoder(
        top_categories=None, variables=["var_A", "var_B"], drop_last=True
    )
    X = encoder.fit_transform(make_df(data_enc_big))

    # test fit attr
    transf = {
        "var_A_A": 6, "var_A_B": 10, "var_A_C": 4, "var_A_D": 10, "var_A_E": 2,
        "var_A_F": 2, "var_B_A": 10, "var_B_B": 6, "var_B_C": 4, "var_B_D": 10,
        "var_B_E": 2, "var_B_F": 2,
    }

    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.variables_binary_ == []
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == {
        "var_A": ["A", "B", "C", "D", "E", "F"],
        "var_B": ["A", "B", "C", "D", "E", "F"],
    }
    # test transform output
    assert isinstance(X, make_df)
    result = frame_to_dict(X)
    assert {col: sum(result[col]) for col in transf} == transf
    assert "var_B" not in result
    assert "var_B_G" not in result
    assert result["var_C"] == data_enc_big["var_C"]


def test_encode_top_categories(make_df, data_enc_top):
    # test case 3: encode only the most popular categories
    encoder = OneHotEncoder(top_categories=4, variables=None, drop_last=False)
    X = encoder.fit_transform(make_df(data_enc_top))

    transf = {
        "var_A_D": 9, "var_A_B": 11, "var_A_A": 5, "var_A_G": 7,
        "var_B_A": 11, "var_B_D": 9, "var_B_G": 5, "var_B_B": 7,
        "var_C_D": 9, "var_C_C": 11, "var_C_G": 7, "var_C_B": 5,
    }

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.variables_binary_ == []
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == {
        "var_A": ["B", "D", "G", "A"],
        "var_B": ["A", "D", "B", "G"],
        "var_C": ["C", "D", "G", "B"],
    }
    # test transform output
    assert isinstance(X, make_df)
    result = frame_to_dict(X)
    assert {col: sum(result[col]) for col in transf} == transf
    assert "var_B" not in result
    assert "var_B_F" not in result


def test_raises_error_if_df_contains_na(make_df, data_enc_big, data_enc_big_na):
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )

    # test case 4: when dataset contains na, fit method
    encoder = OneHotEncoder()
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.fit(make_df(data_enc_big_na))

    # test case 4: when dataset contains na, transform method
    encoder = OneHotEncoder()
    encoder.fit(make_df(data_enc_big))
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(make_df(data_enc_big_na))


def test_encode_numerical_variables(make_df, data_enc_numeric):
    encoder = OneHotEncoder(
        top_categories=None,
        variables=None,
        drop_last=False,
        ignore_format=True,
    )

    X = encoder.fit_transform(make_df(data_enc_numeric)[["var_A", "var_B"]])

    # test fit attr
    transf = {
        "var_A_1": [1] * 6 + [0] * 14,
        "var_A_2": [0] * 6 + [1] * 10 + [0] * 4,
        "var_A_3": [0] * 16 + [1] * 4,
        "var_B_1": [1] * 10 + [0] * 10,
        "var_B_2": [0] * 10 + [1] * 6 + [0] * 4,
        "var_B_3": [0] * 16 + [1] * 4,
    }

    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.variables_binary_ == []
    assert encoder.n_features_in_ == 2
    assert encoder.encoder_dict_ == {"var_A": [1, 2, 3], "var_B": [1, 2, 3]}
    # test transform output
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == transf


def test_variables_cast_as_category(df_enc_numeric):
    # pandas-specific: category dtype has no polars equivalent behavior
    # under test here (encoding categorical-dtype columns).
    df = df_enc_numeric[["var_A", "var_B"]].copy()
    df[["var_A", "var_B"]] = df[["var_A", "var_B"]].astype("category")

    encoder = OneHotEncoder(
        top_categories=None,
        variables=None,
        drop_last=False,
        ignore_format=True,
    )
    X = encoder.fit_transform(df)

    transf = {
        "var_A_1": [1] * 6 + [0] * 14,
        "var_A_2": [0] * 6 + [1] * 10 + [0] * 4,
        "var_A_3": [0] * 16 + [1] * 4,
        "var_B_1": [1] * 10 + [0] * 10,
        "var_B_2": [0] * 10 + [1] * 6 + [0] * 4,
        "var_B_3": [0] * 16 + [1] * 4,
    }

    transf = pd.DataFrame(transf).astype("int32")
    X = pd.DataFrame(X).astype("int32")

    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.n_features_in_ == 2
    assert encoder.encoder_dict_ == {"var_A": [1, 2, 3], "var_B": [1, 2, 3]}
    # test transform output
    pd.testing.assert_frame_equal(X, transf)


def test_encode_into_k_dummy_plus_drop_binary(make_df):
    encoder = OneHotEncoder(
        top_categories=None, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(make_df(DATA_ENC_BINARY))

    # test fit attr
    transf = {
        "var_num": DATA_ENC_BINARY["var_num"],
        "var_A_A": [1] * 6 + [0] * 14,
        "var_A_B": [0] * 6 + [1] * 10 + [0] * 4,
        "var_A_C": [0] * 16 + [1] * 4,
        "var_B_A": [1] * 10 + [0] * 10,
        "var_B_B": [0] * 10 + [1] * 6 + [0] * 4,
        "var_B_C": [0] * 16 + [1] * 4,
        "var_C_AHA": [1] * 12 + [0] * 8,
        "var_D_OHO": [1] * 5 + [0] * 15,
    }

    assert encoder.variables_ == ["var_A", "var_B", "var_C", "var_D"]
    assert encoder.variables_binary_ == ["var_C", "var_D"]
    assert encoder.n_features_in_ == 5
    assert encoder.encoder_dict_ == {
        "var_A": ["A", "B", "C"],
        "var_B": ["A", "B", "C"],
        "var_C": ["AHA"],
        "var_D": ["OHO"],
    }
    # test transform output
    assert isinstance(X, make_df)
    assert list(X.columns) == list(transf)
    assert frame_to_dict(X) == transf


def test_encode_into_kminus1_dummyy_plus_drop_binary(make_df):
    encoder = OneHotEncoder(
        top_categories=None, variables=None, drop_last=True, drop_last_binary=True
    )
    X = encoder.fit_transform(make_df(DATA_ENC_BINARY))

    # test fit attr
    transf = {
        "var_num": DATA_ENC_BINARY["var_num"],
        "var_A_A": [1] * 6 + [0] * 14,
        "var_A_B": [0] * 6 + [1] * 10 + [0] * 4,
        "var_B_A": [1] * 10 + [0] * 10,
        "var_B_B": [0] * 10 + [1] * 6 + [0] * 4,
        "var_C_AHA": [1] * 12 + [0] * 8,
        "var_D_OHO": [1] * 5 + [0] * 15,
    }

    assert encoder.variables_ == ["var_A", "var_B", "var_C", "var_D"]
    assert encoder.variables_binary_ == ["var_C", "var_D"]
    assert encoder.n_features_in_ == 5
    assert encoder.encoder_dict_ == {
        "var_A": ["A", "B"],
        "var_B": ["A", "B"],
        "var_C": ["AHA"],
        "var_D": ["OHO"],
    }
    # test transform output
    assert isinstance(X, make_df)
    assert list(X.columns) == list(transf)
    assert frame_to_dict(X) == transf


def test_encode_into_top_categories_plus_drop_binary(make_df):
    df = make_df(DATA_ENC_BINARY)
    # top_categories = 1
    encoder = OneHotEncoder(
        top_categories=1, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df)

    # test fit attr
    transf = {
        "var_num": DATA_ENC_BINARY["var_num"],
        "var_A_B": [0] * 6 + [1] * 10 + [0] * 4,
        "var_B_A": [1] * 10 + [0] * 10,
        "var_C_AHA": [1] * 12 + [0] * 8,
        "var_D_OHO": [1] * 5 + [0] * 15,
    }

    assert encoder.variables_ == ["var_A", "var_B", "var_C", "var_D"]
    assert encoder.variables_binary_ == ["var_C", "var_D"]
    assert encoder.n_features_in_ == 5
    assert encoder.encoder_dict_ == {
        "var_A": ["B"],
        "var_B": ["A"],
        "var_C": ["AHA"],
        "var_D": ["OHO"],
    }
    # test transform output
    assert isinstance(X, make_df)
    assert list(X.columns) == list(transf)
    assert frame_to_dict(X) == transf

    # top_categories = 2
    encoder = OneHotEncoder(
        top_categories=2, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df)

    # test fit attr
    transf = {
        "var_num": DATA_ENC_BINARY["var_num"],
        "var_A_B": [0] * 6 + [1] * 10 + [0] * 4,
        "var_A_A": [1] * 6 + [0] * 14,
        "var_B_A": [1] * 10 + [0] * 10,
        "var_B_B": [0] * 10 + [1] * 6 + [0] * 4,
        "var_C_AHA": [1] * 12 + [0] * 8,
        "var_D_OHO": [1] * 5 + [0] * 15,
    }

    assert encoder.variables_ == ["var_A", "var_B", "var_C", "var_D"]
    assert encoder.variables_binary_ == ["var_C", "var_D"]
    assert encoder.n_features_in_ == 5
    assert encoder.encoder_dict_ == {
        "var_A": ["B", "A"],
        "var_B": ["A", "B"],
        "var_C": ["AHA"],
        "var_D": ["OHO"],
    }
    # test transform output
    assert isinstance(X, make_df)
    assert list(X.columns) == list(transf)
    assert frame_to_dict(X) == transf


def test_get_feature_names_out(make_df):
    df = make_df(DATA_ENC_BINARY)
    original_features = ["var_num"]
    input_features = list(DATA_ENC_BINARY)

    tr = OneHotEncoder()
    tr.fit(df)

    out = [
        "var_A_A", "var_A_B", "var_A_C", "var_B_A", "var_B_B", "var_B_C",
        "var_C_AHA", "var_C_UHU", "var_D_OHO", "var_D_EHE",
    ]

    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out

    tr = OneHotEncoder(drop_last=True)
    tr.fit(df)

    out = ["var_A_A", "var_A_B", "var_B_A", "var_B_B", "var_C_AHA", "var_D_OHO"]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out

    tr = OneHotEncoder(drop_last_binary=True)
    tr.fit(df)

    out = [
        "var_A_A", "var_A_B", "var_A_C", "var_B_A", "var_B_B", "var_B_C",
        "var_C_AHA", "var_D_OHO",
    ]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out

    tr = OneHotEncoder(top_categories=1)
    tr.fit(df)

    out = ["var_A_B", "var_B_A", "var_C_AHA", "var_D_EHE"]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out

    msg = "input_features must be a list or an array. Got {input_features} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        tr.get_feature_names_out("var_A")

    msg = "input_features is not equal to feature_names_in_"
    with pytest.raises(ValueError, match=re.escape(msg)):
        tr.get_feature_names_out(["var_A", "hola"])


def test_get_feature_names_out_from_pipeline(make_df):
    df = make_df(DATA_ENC_BINARY)
    original_features = ["var_num"]
    input_features = list(DATA_ENC_BINARY)

    tr = Pipeline([("transformer", OneHotEncoder())])
    tr.fit(df)

    out = [
        "var_A_A", "var_A_B", "var_A_C", "var_B_A", "var_B_B", "var_B_C",
        "var_C_AHA", "var_C_UHU", "var_D_OHO", "var_D_EHE",
    ]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out


def test_inverse_transform_raises_not_implemented_error(make_df):
    df = make_df(DATA_ENC_BINARY)
    enc = OneHotEncoder().fit(df)
    msg = "inverse_transform is not implemented for this transformer."
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        enc.inverse_transform(df)
