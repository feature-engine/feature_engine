from difflib import SequenceMatcher

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.encoding import StringSimilarityEncoder
from feature_engine.encoding.similarity_encoder import _gpm_fast

DATA_ENC = {
    "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}
DATA_ENC_BIG = {
    "var_A": ["A"] * 6
    + ["B"] * 10
    + ["C"] * 4
    + ["D"] * 10
    + ["E"] * 2
    + ["F"] * 2
    + ["G"] * 6,
    "var_B": ["A"] * 10
    + ["B"] * 6
    + ["C"] * 4
    + ["D"] * 10
    + ["E"] * 2
    + ["F"] * 2
    + ["G"] * 6,
    "var_C": ["A"] * 4
    + ["B"] * 6
    + ["C"] * 10
    + ["D"] * 10
    + ["E"] * 2
    + ["F"] * 2
    + ["G"] * 6,
}
# only var_A carries the null (matches the original single-column NA fixture)
DATA_ENC_BIG_NA = {**DATA_ENC_BIG, "var_A": [None] + DATA_ENC_BIG["var_A"][1:]}

DATA_ENC_TOP = {
    "var_A": ["A"] * 5
    + ["B"] * 11
    + ["C"] * 4
    + ["D"] * 9
    + ["E"] * 2
    + ["F"] * 2
    + ["G"] * 7,
    "var_B": ["A"] * 11
    + ["B"] * 7
    + ["C"] * 4
    + ["D"] * 9
    + ["E"] * 2
    + ["F"] * 2
    + ["G"] * 5,
    "var_C": ["A"] * 4
    + ["B"] * 5
    + ["C"] * 11
    + ["D"] * 9
    + ["E"] * 2
    + ["F"] * 2
    + ["G"] * 7,
}


def _to_pandas(X):
    return nw.from_native(X, eager_only=True).to_pandas()


def _columns(X):
    return list(nw.from_native(X, eager_only=True).columns)


@pytest.mark.parametrize(
    "strings", [("hola", "chau"), ("hi there", "hi here"), (100, 1000)]
)
def test_gpm_fast(strings):
    str1, str2 = strings
    assert SequenceMatcher(None, str(str1), str(str2)).quick_ratio() == _gpm_fast(
        str1, str2
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_top_categories(make_df):
    df = make_df(DATA_ENC_TOP)

    encoder = StringSimilarityEncoder(top_categories=4)
    X = encoder.fit_transform(df)

    # test init params
    assert encoder.top_categories == 4
    # test fit attr
    transf = {
        "var_A_D": 9,
        "var_A_B": 11,
        "var_A_A": 5,
        "var_A_G": 7,
        "var_B_A": 11,
        "var_B_D": 9,
        "var_B_G": 5,
        "var_B_B": 7,
        "var_C_D": 9,
        "var_C_C": 11,
        "var_C_G": 7,
        "var_C_B": 5,
    }

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == {
        "var_A": ["B", "D", "G", "A"],
        "var_B": ["A", "D", "B", "G"],
        "var_C": ["C", "D", "G", "B"],
    }
    # test transform output
    for col in transf.keys():
        assert X[col].sum() == transf[col]
    assert "var_B" not in _columns(X)
    assert "var_B_F" not in _columns(X)


@pytest.mark.parametrize("top_cat", ["hello", 0.5, [1]])
def test_error_if_top_categories_not_integer(top_cat):
    with pytest.raises(ValueError):
        StringSimilarityEncoder(top_categories=top_cat)


@pytest.mark.parametrize(
    "handle_missing", ["error", "propagate", ["raise"], 1, 0.1, False]
)
def test_error_if_handle_missing_invalid(handle_missing):
    with pytest.raises(ValueError):
        StringSimilarityEncoder(missing_values=handle_missing)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("missing_vals", ["other", False, 1])
def test_error_if_missing_values_not_recognized_in_fit(missing_vals, make_df):
    df_enc = make_df(DATA_ENC)
    enc = StringSimilarityEncoder()
    enc.missing_values = missing_vals
    with pytest.raises(ValueError):
        enc.fit(df_enc)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_nan_behaviour_error_fit(make_df):
    df_enc_big_na = make_df(DATA_ENC_BIG_NA)
    encoder = StringSimilarityEncoder(missing_values="raise")
    with pytest.raises(ValueError, match=(
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )):
        encoder.fit(df_enc_big_na)


# pandas offers several NA sentinels (np.nan, pd.NA, None); polars only has
# a single null representation, so this stays pandas-only.
@pytest.mark.parametrize("nan_value", [np.nan, pd.NA, None])
def test_nan_behaviour_error_transform(nan_value):
    df_enc_big = pd.DataFrame(DATA_ENC_BIG)
    encoder = StringSimilarityEncoder(missing_values="raise")
    encoder.fit(df_enc_big)

    df_enc_big_na = df_enc_big.copy()
    df_enc_big_na.loc[0, "var_A"] = nan_value

    with pytest.raises(ValueError, match=(
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )):
        encoder.transform(df_enc_big_na)


@pytest.mark.parametrize("nan_value", [np.nan, pd.NA, None])
def test_nan_behaviour_impute(nan_value):
    df_enc_big_na = pd.DataFrame(DATA_ENC_BIG)
    df_enc_big_na.loc[0, "var_A"] = nan_value

    encoder = StringSimilarityEncoder(missing_values="impute")
    X = encoder.fit_transform(df_enc_big_na)

    assert (X.isna().sum() == 0).all(axis=None)
    assert encoder.encoder_dict_ == {
        "var_A": ["B", "D", "G", "A", "C", "E", "F", ""],
        "var_B": ["A", "D", "B", "G", "C", "E", "F"],
        "var_C": ["C", "D", "B", "G", "A", "E", "F"],
    }


@pytest.mark.parametrize("nan_value", [np.nan, pd.NA, None])
def test_nan_behaviour_ignore(nan_value):
    df_enc_big_na = pd.DataFrame(DATA_ENC_BIG)
    df_enc_big_na.loc[0, "var_A"] = nan_value

    encoder = StringSimilarityEncoder(missing_values="ignore")
    X = encoder.fit_transform(df_enc_big_na)
    assert (X.isna().any(axis=1) == df_enc_big_na.isna().any(axis=1)).all()
    assert encoder.encoder_dict_ == {
        "var_A": ["B", "D", "G", "A", "C", "E", "F"],
        "var_B": ["A", "D", "B", "G", "C", "E", "F"],
        "var_C": ["C", "D", "B", "G", "A", "E", "F"],
    }


def test_string_dtype_with_pd_na():
    # pandas nullable "string" dtype is pandas-specific.
    df = pd.DataFrame({"var_A": ["A", "B", pd.NA]}, dtype="string")
    encoder = StringSimilarityEncoder(missing_values="impute")
    X = encoder.fit_transform(df)
    assert (X.isna().sum() == 0).all(axis=None)
    assert "" in encoder.encoder_dict_["var_A"]


def test_string_dtype_with_literal_nan_strings():
    # literal "nan"/"<NA>" strings (not real nulls) must be treated as
    # ordinary categories; pandas nullable "string" dtype is pandas-specific.
    df = pd.DataFrame({"var_A": ["nan", "<NA>", "A", "B"]}, dtype="string")
    encoder = StringSimilarityEncoder(missing_values="impute")
    X = encoder.fit_transform(df)
    assert (X.isna().sum() == 0).all(axis=None)
    assert "nan" in encoder.encoder_dict_["var_A"]
    assert "<NA>" in encoder.encoder_dict_["var_A"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_error(make_df):
    df_enc_big = make_df(DATA_ENC_BIG)
    encoder = StringSimilarityEncoder()
    X = encoder.fit_transform(df_enc_big)
    with pytest.raises(NotImplementedError):
        encoder.inverse_transform(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out(make_df):
    df_enc_big = make_df(DATA_ENC_BIG)
    input_features = _columns(df_enc_big)

    tr = StringSimilarityEncoder()
    tr.fit(df_enc_big)

    # sort by popularity within variable
    out = [
        "var_A_B",
        "var_A_D",
        "var_A_A",
        "var_A_G",
        "var_A_C",
        "var_A_E",
        "var_A_F",
        "var_B_A",
        "var_B_D",
        "var_B_B",
        "var_B_G",
        "var_B_C",
        "var_B_E",
        "var_B_F",
        "var_C_C",
        "var_C_D",
        "var_C_B",
        "var_C_G",
        "var_C_A",
        "var_C_E",
        "var_C_F",
    ]

    assert tr.get_feature_names_out(input_features=None) == out
    assert tr.get_feature_names_out(input_features=input_features) == out

    tr = StringSimilarityEncoder(top_categories=1)
    tr.fit(df_enc_big)

    out = ["var_A_B", "var_B_A", "var_C_C"]

    assert tr.get_feature_names_out(input_features=None) == out
    assert tr.get_feature_names_out(input_features=input_features) == out

    with pytest.raises(ValueError):
        tr.get_feature_names_out("var_A")

    with pytest.raises(ValueError):
        tr.get_feature_names_out(["var_A", "hola"])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out_na(make_df):
    df_enc_big_na = make_df(DATA_ENC_BIG_NA)
    input_features = _columns(df_enc_big_na)

    tr = StringSimilarityEncoder()
    tr.fit(df_enc_big_na)

    out = [
        "var_A_B",
        "var_A_D",
        "var_A_G",
        "var_A_A",
        "var_A_C",
        "var_A_E",
        "var_A_F",
        "var_A_nan",
        "var_B_A",
        "var_B_D",
        "var_B_B",
        "var_B_G",
        "var_B_C",
        "var_B_E",
        "var_B_F",
        "var_C_C",
        "var_C_D",
        "var_C_B",
        "var_C_G",
        "var_C_A",
        "var_C_E",
        "var_C_F",
    ]

    # NaN values are replaced with empty string "" before string conversion
    assert tr.encoder_dict_ == {
        "var_A": ["B", "D", "G", "A", "C", "E", "F", ""],
        "var_B": ["A", "D", "B", "G", "C", "E", "F"],
        "var_C": ["C", "D", "B", "G", "A", "E", "F"],
    }
    assert tr.get_feature_names_out(input_features=None) == out
    assert tr.get_feature_names_out(input_features=input_features) == out


@pytest.mark.parametrize("keywords", ["hello", 0.5, [1]])
def test_keywords_bad_type(keywords):
    with pytest.raises(ValueError):
        StringSimilarityEncoder(keywords=keywords)


@pytest.mark.parametrize("item", ["hello", 0.5, 1])
def test_keywords_bad_items(item):
    with pytest.raises(ValueError):
        StringSimilarityEncoder(keywords={"var_A": item})


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("key", ["hello", 0.5, 1])
def test_keywords_bad_keys(key, make_df):
    df_enc_big = make_df(DATA_ENC_BIG)
    encoder = StringSimilarityEncoder(keywords={key: ["A"]})
    with pytest.raises(ValueError):
        encoder.fit(df_enc_big)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_partial_keywords(make_df):
    df = make_df(DATA_ENC_TOP)

    encoder = StringSimilarityEncoder(top_categories=2, keywords={"var_A": ["XYZ"]})
    X = encoder.fit_transform(df)

    # test init params
    assert encoder.top_categories == 2
    # test fit attr
    transf = {
        "var_A_XYZ": 0,
        "var_B_A": 11,
        "var_B_D": 9,
        "var_C_D": 9,
        "var_C_C": 11,
    }

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == {
        "var_A": ["XYZ"],
        "var_B": ["A", "D"],
        "var_C": ["C", "D"],
    }
    # test transform output
    for col in transf.keys():
        assert X[col].sum() == transf[col]
    assert "var_B" not in _columns(X)
    assert "var_B_F" not in _columns(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_complete_keywords(make_df):
    df = make_df(DATA_ENC_TOP)

    encoder = StringSimilarityEncoder(
        keywords={"var_A": ["X"], "var_B": ["Y"], "var_C": ["Z"]}
    )
    X = encoder.fit_transform(df)

    # test fit attr
    transf = {
        "var_A_X": 0,
        "var_B_Y": 0,
        "var_C_Z": 0,
    }

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == {
        "var_A": ["X"],
        "var_B": ["Y"],
        "var_C": ["Z"],
    }
    # test transform output
    for col in transf.keys():
        assert X[col].sum() == transf[col]
    assert "var_B" not in _columns(X)
    assert "var_B_F" not in _columns(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out_w_keywords(make_df):
    df_enc_big_na = make_df(DATA_ENC_BIG_NA)
    input_features = _columns(df_enc_big_na)

    tr = StringSimilarityEncoder(keywords={"var_A": ["XYZ"]})
    tr.fit(df_enc_big_na)

    out = [
        "var_A_XYZ",
        "var_B_A",
        "var_B_D",
        "var_B_B",
        "var_B_G",
        "var_B_C",
        "var_B_E",
        "var_B_F",
        "var_C_C",
        "var_C_D",
        "var_C_B",
        "var_C_G",
        "var_C_A",
        "var_C_E",
        "var_C_F",
    ]

    assert tr.encoder_dict_ == {
        "var_A": ["XYZ"],
        "var_B": ["A", "D", "B", "G", "C", "E", "F"],
        "var_C": ["C", "D", "B", "G", "A", "E", "F"],
    }
    assert tr.get_feature_names_out(input_features=None) == out
    assert tr.get_feature_names_out(input_features=input_features) == out
