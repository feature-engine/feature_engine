from difflib import SequenceMatcher

import pandas as pd
import pytest

from feature_engine.encoding import StringSimilarityEncoder
from feature_engine.encoding.similarity_encoder import _gpm_fast


@pytest.mark.parametrize(
    "strings", [("hola", "chau"), ("hi there", "hi here"), (100, 1000)]
)
def test_gpm_fast(strings):
    str1, str2 = strings
    assert SequenceMatcher(None, str(str1), str(str2)).quick_ratio() == _gpm_fast(
        str1, str2
    )


def test_encode_top_categories():
    df = pd.DataFrame(
        {
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
    )

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
    assert "var_B" not in X.columns
    assert "var_B_F" not in X.columns


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


@pytest.mark.parametrize("missing_vals", ["other", False, 1])
def test_error_if_missing_values_not_recognized_in_fit(missing_vals, df_enc):
    enc = StringSimilarityEncoder()
    enc.missing_values = missing_vals
    with pytest.raises(ValueError):
        enc.fit(df_enc)


def test_nan_behaviour_error_fit(df_enc_big_na):
    encoder = StringSimilarityEncoder(missing_values="raise")
    with pytest.raises(ValueError) as record:
        encoder.fit(df_enc_big_na)

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_nan_behaviour_error_transform(df_enc_big, df_enc_big_na):
    encoder = StringSimilarityEncoder(missing_values="raise")
    encoder.fit(df_enc_big)
    with pytest.raises(ValueError) as record:
        encoder.transform(df_enc_big_na)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_nan_behaviour_impute(df_enc_big_na):
    encoder = StringSimilarityEncoder(missing_values="impute")
    X = encoder.fit_transform(df_enc_big_na)
    assert (X.isna().sum() == 0).all(axis=None)
    assert encoder.encoder_dict_ == {
        "var_A": ["B", "D", "G", "A", "C", "E", "F", ""],
        "var_B": ["A", "D", "B", "G", "C", "E", "F"],
        "var_C": ["C", "D", "B", "G", "A", "E", "F"],
    }


def test_nan_behaviour_ignore(df_enc_big_na):
    encoder = StringSimilarityEncoder(missing_values="ignore")
    X = encoder.fit_transform(df_enc_big_na)
    assert (X.isna().any(axis=1) == df_enc_big_na.isna().any(axis=1)).all()
    assert encoder.encoder_dict_ == {
        "var_A": ["B", "D", "G", "A", "C", "E", "F"],
        "var_B": ["A", "D", "B", "G", "C", "E", "F"],
        "var_C": ["C", "D", "B", "G", "A", "E", "F"],
    }


def test_inverse_transform_error(df_enc_big):
    encoder = StringSimilarityEncoder()
    X = encoder.fit_transform(df_enc_big)
    with pytest.raises(NotImplementedError):
        encoder.inverse_transform(X)


def test_get_feature_names_out(df_enc_big):
    input_features = df_enc_big.columns.tolist()

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


def test_get_feature_names_out_na(df_enc_big_na):
    input_features = df_enc_big_na.columns.tolist()

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


@pytest.mark.parametrize("key", ["hello", 0.5, 1])
def test_keywords_bad_keys(df_enc_big, key):
    encoder = StringSimilarityEncoder(keywords={key: ["A"]})
    with pytest.raises(ValueError):
        encoder.fit(df_enc_big)


def test_encode_partial_keywords():
    df = pd.DataFrame(
        {
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
    )

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
    assert "var_B" not in X.columns
    assert "var_B_F" not in X.columns


def test_encode_complete_keywords():
    df = pd.DataFrame(
        {
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
    )

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
    assert "var_B" not in X.columns
    assert "var_B_F" not in X.columns


def test_get_feature_names_out_w_keywords(df_enc_big_na):
    input_features = df_enc_big_na.columns.tolist()

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
