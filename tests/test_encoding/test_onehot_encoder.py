import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.encoding import OneHotEncoder


@pytest.mark.parametrize("index_", [[1, 2, 3], [3, 2, 1], [4, 9, 2]])
def test_concat_with_non_ordered_index(index_):
    df = pd.DataFrame({"varA": ["a", "b", "c"], "varB": ["d", "d", "a"]}, index=index_)
    encoder = OneHotEncoder()
    dft = encoder.fit_transform(df)
    df_expected = pd.DataFrame(
        {
            "varA_a": [1, 0, 0],
            "varA_b": [0, 1, 0],
            "varA_c": [0, 0, 1],
            "varB_d": [1, 1, 0],
            "varB_a": [0, 0, 1],
        },
        index=index_,
    )
    pd.testing.assert_frame_equal(dft, df_expected, check_dtype=False)


def test_encode_categories_in_k_binary_plus_select_vars_automatically(df_enc_big):
    # test case 1: encode all categories into k binary variables, select variables
    # automatically
    encoder = OneHotEncoder(top_categories=None, variables=None, drop_last=False)
    X = encoder.fit_transform(df_enc_big)

    # test init params
    assert encoder.top_categories is None
    assert encoder.variables is None
    assert encoder.drop_last is False
    # test fit attr
    transf = {
        "var_A_A": 6,
        "var_A_B": 10,
        "var_A_C": 4,
        "var_A_D": 10,
        "var_A_E": 2,
        "var_A_F": 2,
        "var_A_G": 6,
        "var_B_A": 10,
        "var_B_B": 6,
        "var_B_C": 4,
        "var_B_D": 10,
        "var_B_E": 2,
        "var_B_F": 2,
        "var_B_G": 6,
        "var_C_A": 4,
        "var_C_B": 6,
        "var_C_C": 10,
        "var_C_D": 10,
        "var_C_E": 2,
        "var_C_F": 2,
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
    assert X.sum().to_dict() == transf
    assert "var_A" not in X.columns


def test_encode_categories_in_k_minus_1_binary_plus_list_of_variables(df_enc_big):
    # test case 2: encode all categories into k-1 binary variables,
    # pass list of variables
    encoder = OneHotEncoder(
        top_categories=None, variables=["var_A", "var_B"], drop_last=True
    )
    X = encoder.fit_transform(df_enc_big)

    # test init params
    assert encoder.top_categories is None
    assert encoder.variables == ["var_A", "var_B"]
    assert encoder.drop_last is True
    # test fit attr
    transf = {
        "var_A_A": 6,
        "var_A_B": 10,
        "var_A_C": 4,
        "var_A_D": 10,
        "var_A_E": 2,
        "var_A_F": 2,
        "var_B_A": 10,
        "var_B_B": 6,
        "var_B_C": 4,
        "var_B_D": 10,
        "var_B_E": 2,
        "var_B_F": 2,
    }

    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.variables_binary_ == []
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == {
        "var_A": ["A", "B", "C", "D", "E", "F"],
        "var_B": ["A", "B", "C", "D", "E", "F"],
    }
    # test transform output
    for col in transf.keys():
        assert X[col].sum() == transf[col]
    assert "var_B" not in X.columns
    assert "var_B_G" not in X.columns
    assert "var_C" in X.columns


def test_encode_top_categories():
    # test case 3: encode only the most popular categories

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

    encoder = OneHotEncoder(top_categories=4, variables=None, drop_last=False)
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
    assert encoder.variables_binary_ == []
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


# init params
@pytest.mark.parametrize("top_cat", ["empanada", [1], 0.5, -1])
def test_error_if_top_categories_not_integer(top_cat):
    with pytest.raises(ValueError):
        OneHotEncoder(top_categories=top_cat)


@pytest.mark.parametrize("drop_last", ["empanada", [1], 0.5, -1, 1])
def test_error_if_drop_last_not_bool(drop_last):
    with pytest.raises(ValueError):
        OneHotEncoder(drop_last=drop_last)


@pytest.mark.parametrize("drop_binary", ["hello", ["auto"], -1, 100, 0.5])
def test_raises_error_when_not_allowed_smoothing_param_in_init(drop_binary):
    with pytest.raises(ValueError):
        OneHotEncoder(drop_last_binary=drop_binary)


def test_raises_error_if_df_contains_na(df_enc_big, df_enc_big_na):
    # test case 4: when dataset contains na, fit method
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )

    encoder = OneHotEncoder()
    with pytest.raises(ValueError) as record:
        encoder.fit(df_enc_big_na)

    assert str(record.value) == msg

    # test case 4: when dataset contains na, transform method
    encoder = OneHotEncoder()
    encoder.fit(df_enc_big)
    with pytest.raises(ValueError):
        encoder.transform(df_enc_big_na)
    assert str(record.value) == msg


def test_encode_numerical_variables(df_enc_numeric):
    encoder = OneHotEncoder(
        top_categories=None,
        variables=None,
        drop_last=False,
        ignore_format=True,
    )

    X = encoder.fit_transform(df_enc_numeric[["var_A", "var_B"]])

    # test fit attr
    transf = {
        "var_A_1": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_A_2": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_A_3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "var_B_1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    }

    transf = pd.DataFrame(transf).astype("int32")
    X = pd.DataFrame(X).astype("int32")

    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.variables_binary_ == []
    assert encoder.n_features_in_ == 2
    assert encoder.encoder_dict_ == {"var_A": [1, 2, 3], "var_B": [1, 2, 3]}
    # test transform output
    pd.testing.assert_frame_equal(X, transf)


def test_variables_cast_as_category(df_enc_numeric):
    encoder = OneHotEncoder(
        top_categories=None,
        variables=None,
        drop_last=False,
        ignore_format=True,
    )

    df = df_enc_numeric.copy()
    df[["var_A", "var_B"]] = df[["var_A", "var_B"]].astype("category")

    X = encoder.fit_transform(df[["var_A", "var_B"]])

    # test fit attr
    transf = {
        "var_A_1": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_A_2": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_A_3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "var_B_1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    }

    transf = pd.DataFrame(transf).astype("int32")
    X = pd.DataFrame(X).astype("int32")

    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.n_features_in_ == 2
    assert encoder.encoder_dict_ == {"var_A": [1, 2, 3], "var_B": [1, 2, 3]}
    # test transform output
    pd.testing.assert_frame_equal(X, transf)


@pytest.fixture(scope="module")
def df_enc_binary():
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_C": ["AHA"] * 12 + ["UHU"] * 8,
        "var_D": ["OHO"] * 5 + ["EHE"] * 15,
        "var_num": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)

    return df


def test_encode_into_k_dummy_plus_drop_binary(df_enc_binary):
    encoder = OneHotEncoder(
        top_categories=None, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df_enc_binary)
    X = X.astype("int32")

    # test fit attr
    transf = {
        "var_num": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_A_C": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_B": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_C": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "var_C_AHA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_D_OHO": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    transf = pd.DataFrame(transf).astype("int32")

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
    pd.testing.assert_frame_equal(X, transf)
    assert "var_C_B" not in X.columns


def test_encode_into_kminus1_dummyy_plus_drop_binary(df_enc_binary):
    encoder = OneHotEncoder(
        top_categories=None, variables=None, drop_last=True, drop_last_binary=True
    )
    X = encoder.fit_transform(df_enc_binary)
    X = X.astype("int32")

    # test fit attr
    transf = {
        "var_num": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_B": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_C_AHA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_D_OHO": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    transf = pd.DataFrame(transf).astype("int32")

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
    pd.testing.assert_frame_equal(X, transf)
    assert "var_C_B" not in X.columns


def test_encode_into_top_categories_plus_drop_binary(df_enc_binary):
    # top_categories = 1
    encoder = OneHotEncoder(
        top_categories=1, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df_enc_binary)
    X = X.astype("int32")

    # test fit attr
    transf = {
        "var_num": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_C_AHA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_D_OHO": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    transf = pd.DataFrame(transf).astype("int32")

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
    pd.testing.assert_frame_equal(X, transf)
    assert "var_C_B" not in X.columns

    # top_categories = 2
    encoder = OneHotEncoder(
        top_categories=2, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df_enc_binary)
    X = X.astype("int32")

    # test fit attr
    transf = {
        "var_num": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_A_A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_B": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_C_AHA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_D_OHO": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    transf = pd.DataFrame(transf).astype("int32")

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
    pd.testing.assert_frame_equal(X, transf)
    assert "var_C_B" not in X.columns


def test_get_feature_names_out(df_enc_binary):
    original_features = ["var_num"]
    input_features = df_enc_binary.columns

    tr = OneHotEncoder()
    tr.fit(df_enc_binary)

    out = [
        "var_A_A",
        "var_A_B",
        "var_A_C",
        "var_B_A",
        "var_B_B",
        "var_B_C",
        "var_C_AHA",
        "var_C_UHU",
        "var_D_OHO",
        "var_D_EHE",
    ]

    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out

    tr = OneHotEncoder(drop_last=True)
    tr.fit(df_enc_binary)

    out = [
        "var_A_A",
        "var_A_B",
        "var_B_A",
        "var_B_B",
        "var_C_AHA",
        "var_D_OHO",
    ]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out

    tr = OneHotEncoder(drop_last_binary=True)
    tr.fit(df_enc_binary)

    out = [
        "var_A_A",
        "var_A_B",
        "var_A_C",
        "var_B_A",
        "var_B_B",
        "var_B_C",
        "var_C_AHA",
        "var_D_OHO",
    ]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out

    tr = OneHotEncoder(top_categories=1)
    tr.fit(df_enc_binary)

    out = ["var_A_B", "var_B_A", "var_C_AHA", "var_D_EHE"]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out

    with pytest.raises(ValueError):
        tr.get_feature_names_out("var_A")

    with pytest.raises(ValueError):
        tr.get_feature_names_out(["var_A", "hola"])


def test_get_feature_names_out_from_pipeline(df_enc_binary):
    original_features = ["var_num"]
    input_features = df_enc_binary.columns

    tr = Pipeline([("transformer", OneHotEncoder())])
    tr.fit(df_enc_binary)

    out = [
        "var_A_A",
        "var_A_B",
        "var_A_C",
        "var_B_A",
        "var_B_B",
        "var_B_C",
        "var_C_AHA",
        "var_C_UHU",
        "var_D_OHO",
        "var_D_EHE",
    ]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out


def test_inverse_transform_raises_not_implemented_error(df_enc_binary):
    enc = OneHotEncoder().fit(df_enc_binary)
    with pytest.raises(NotImplementedError):
        enc.inverse_transform(df_enc_binary)


# ===========================================================================
# Tests for the new `drop` parameter (Issue #913)
# ===========================================================================


@pytest.fixture(scope="module")
def df_drop():
    """DataFrame with known categories for testing drop strategies."""
    df = pd.DataFrame(
        {
            "x1": ["c", "a", "b", "a", "c", "b", "a"],
            "x2": ["z", "y", "z", "x", "y", "z", "x"],
            "num": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    return df


def test_drop_last_alphabetically(df_drop):
    """drop='last' should drop the last category in sorted order."""
    encoder = OneHotEncoder(drop="last")
    encoder.fit(df_drop)

    # x1 categories sorted: ['a', 'b', 'c'] -> drop 'c'
    assert encoder.encoder_dict_["x1"] == ["a", "b"]
    # x2 categories sorted: ['x', 'y', 'z'] -> drop 'z'
    assert encoder.encoder_dict_["x2"] == ["x", "y"]

    X = encoder.transform(df_drop)
    assert "x1_c" not in X.columns
    assert "x2_z" not in X.columns
    assert "x1_a" in X.columns
    assert "x1_b" in X.columns
    assert "x2_x" in X.columns
    assert "x2_y" in X.columns


def test_drop_first_alphabetically(df_drop):
    """drop='first' should drop the first category in sorted order."""
    encoder = OneHotEncoder(drop="first")
    encoder.fit(df_drop)

    # x1 categories sorted: ['a', 'b', 'c'] -> drop 'a'
    assert encoder.encoder_dict_["x1"] == ["b", "c"]
    # x2 categories sorted: ['x', 'y', 'z'] -> drop 'x'
    assert encoder.encoder_dict_["x2"] == ["y", "z"]

    X = encoder.transform(df_drop)
    assert "x1_a" not in X.columns
    assert "x2_x" not in X.columns
    assert "x1_b" in X.columns
    assert "x1_c" in X.columns
    assert "x2_y" in X.columns
    assert "x2_z" in X.columns


def test_drop_most_frequent():
    """drop='most_frequent' should drop the most common category."""
    df = pd.DataFrame(
        {
            "x1": ["a"] * 10 + ["b"] * 5 + ["c"] * 3,
        }
    )

    encoder = OneHotEncoder(drop="most_frequent")
    encoder.fit(df)

    # 'a' is most frequent (10 times) -> drop 'a'
    assert "a" not in encoder.encoder_dict_["x1"]
    assert "b" in encoder.encoder_dict_["x1"]
    assert "c" in encoder.encoder_dict_["x1"]

    X = encoder.transform(df)
    assert "x1_a" not in X.columns
    assert "x1_b" in X.columns
    assert "x1_c" in X.columns


def test_drop_most_frequent_with_tie():
    """When multiple categories tie for most frequent, warn and drop first alpha."""
    df = pd.DataFrame(
        {
            "x1": ["c"] * 5 + ["a"] * 5 + ["b"] * 3,
        }
    )

    with pytest.warns(UserWarning, match="multiple categories share the highest"):
        encoder = OneHotEncoder(drop="most_frequent")
        encoder.fit(df)

    # 'a' and 'c' both have frequency 5 — drop 'a' (first alphabetically)
    assert "a" not in encoder.encoder_dict_["x1"]
    assert "b" in encoder.encoder_dict_["x1"]
    assert "c" in encoder.encoder_dict_["x1"]


def test_drop_ignored_when_top_categories_set():
    """top_categories should take precedence over drop."""
    df = pd.DataFrame(
        {
            "x1": ["a"] * 10 + ["b"] * 5 + ["c"] * 3 + ["d"] * 1,
        }
    )

    encoder = OneHotEncoder(top_categories=2, drop="first")
    encoder.fit(df)

    # top_categories=2 should pick the 2 most frequent: ['a', 'b']
    assert encoder.encoder_dict_["x1"] == ["a", "b"]


def test_drop_overrides_drop_last():
    """When both drop and drop_last are set, drop wins and FutureWarning is raised."""
    df = pd.DataFrame(
        {
            "x1": ["c", "a", "b", "a", "c", "b", "a"],
        }
    )

    with pytest.warns(FutureWarning, match="drop_last.*deprecated"):
        encoder = OneHotEncoder(drop_last=True, drop="first")

    encoder.fit(df)

    # drop="first" should drop 'a' (sorted: ['a', 'b', 'c'])
    assert encoder.encoder_dict_["x1"] == ["b", "c"]


def test_drop_with_drop_last_binary():
    """drop and drop_last_binary should work together correctly."""
    df = pd.DataFrame(
        {
            "x1": ["a"] * 10 + ["b"] * 5 + ["c"] * 3,
            "x2": ["yes"] * 10 + ["no"] * 8,  # binary variable
        }
    )

    encoder = OneHotEncoder(drop="first", drop_last_binary=True)
    encoder.fit(df)

    # x1: sorted ['a', 'b', 'c'] -> drop 'a'
    assert encoder.encoder_dict_["x1"] == ["b", "c"]

    # x2: binary -> drop_last_binary overrides to keep only the first unique
    assert len(encoder.encoder_dict_["x2"]) == 1


@pytest.mark.parametrize(
    "drop_value", ["empanada", "middle", 123, True, ["last"]]
)
def test_error_if_drop_not_valid_string(drop_value):
    """Invalid drop values should raise ValueError."""
    with pytest.raises(ValueError, match="drop takes only values"):
        OneHotEncoder(drop=drop_value)


def test_get_feature_names_out_with_drop(df_enc_binary):
    """get_feature_names_out should reflect the dropped category."""
    original_features = ["var_num"]
    input_features = df_enc_binary.columns

    # drop="first": sorted cats for var_A are ['A','B','C'] -> drop 'A'
    tr = OneHotEncoder(drop="first")
    tr.fit(df_enc_binary)

    out = [
        "var_A_B",
        "var_A_C",
        "var_B_B",
        "var_B_C",
        "var_C_UHU",
        "var_D_OHO",
    ]
    feat_out = original_features + out
    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out


def test_drop_none_produces_k_dummies(df_drop):
    """drop=None (default) should produce k dummies, same as drop_last=False."""
    encoder = OneHotEncoder(drop=None, drop_last=False)
    encoder.fit(df_drop)

    # x1 has 3 unique categories -> 3 dummies
    assert len(encoder.encoder_dict_["x1"]) == 3
    # x2 has 3 unique categories -> 3 dummies
    assert len(encoder.encoder_dict_["x2"]) == 3


def test_drop_last_backward_compatible(df_drop):
    """Existing drop_last=True without drop should behave exactly as before."""
    encoder = OneHotEncoder(drop_last=True)
    encoder.fit(df_drop)

    # Original behavior: category_ls = list(unique()), drop last element
    # This preserves insertion order, NOT sorted order
    x1_unique = list(df_drop["x1"].unique())
    assert encoder.encoder_dict_["x1"] == x1_unique[:-1]

    x2_unique = list(df_drop["x2"].unique())
    assert encoder.encoder_dict_["x2"] == x2_unique[:-1]
