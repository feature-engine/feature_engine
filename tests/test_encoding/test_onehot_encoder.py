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
