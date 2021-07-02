import pandas as pd
import pytest

from feature_engine.encoding import OneHotEncoder


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
    assert encoder.n_features_in_ == 3
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
    assert encoder.n_features_in_ == 3
    # test transform output
    for col in transf.keys():
        assert X[col].sum() == transf[col]
    assert "var_B" not in X.columns
    assert "var_B_G" not in X.columns
    assert "var_C" in X.columns


def test_encode_top_categories(df_enc_big):
    # test case 3: encode only the most popular categories
    encoder = OneHotEncoder(top_categories=4, variables=None, drop_last=False)
    X = encoder.fit_transform(df_enc_big)

    # test init params
    assert encoder.top_categories == 4
    # test fit attr
    transf = {
        "var_A_D": 10,
        "var_A_B": 10,
        "var_A_A": 6,
        "var_A_G": 6,
        "var_B_A": 10,
        "var_B_D": 10,
        "var_B_G": 6,
        "var_B_B": 6,
        "var_C_D": 10,
        "var_C_C": 10,
        "var_C_G": 6,
        "var_C_B": 6,
    }

    assert encoder.n_features_in_ == 3
    # test transform output
    for col in transf.keys():
        assert X[col].sum() == transf[col]
    assert "var_B" not in X.columns
    assert "var_B_F" not in X.columns


def test_error_if_top_categories_not_integer():
    with pytest.raises(ValueError):
        OneHotEncoder(top_categories=0.5)


def test_error_if_drop_last_not_bool():
    with pytest.raises(ValueError):
        OneHotEncoder(drop_last=0.5)


def test_fit_raises_error_if_df_contains_na(df_enc_big_na):
    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = OneHotEncoder()
        encoder.fit(df_enc_big_na)


def test_transform_raises_error_if_df_contains_na(df_enc_big, df_enc_big_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = OneHotEncoder()
        encoder.fit(df_enc_big)
        encoder.transform(df_enc_big_na)


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
    assert encoder.n_features_in_ == 2
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
    # test transform output
    pd.testing.assert_frame_equal(X, transf)


@pytest.fixture(scope="module")
def df_enc_binary():
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_C": ["A"] * 10 + ["B"] * 10,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)

    return df


def test_encode_into_k_binary_plus_drop_binary(df_enc_binary):
    encoder = OneHotEncoder(
        top_categories=None, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df_enc_binary)

    # test fit attr
    transf = {
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_A_C": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_B": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_C": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "var_C_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    transf = pd.DataFrame(transf).astype("int32")
    X = pd.DataFrame(X).astype("int32")

    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 4
    # test transform output
    pd.testing.assert_frame_equal(X, transf)
    assert "var_C_B" not in X.columns


def test_encode_into_kminus1_binary_plus_drop_binary(df_enc_binary):
    encoder = OneHotEncoder(
        top_categories=None, variables=None, drop_last=True, drop_last_binary=True
    )
    X = encoder.fit_transform(df_enc_binary)

    # test fit attr
    transf = {
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_B": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_C_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    transf = pd.DataFrame(transf).astype("int32")
    X = pd.DataFrame(X).astype("int32")

    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 4
    # test transform output
    pd.testing.assert_frame_equal(X, transf)
    assert "var_C_B" not in X.columns


def test_encode_into_top_categories_plus_drop_binary(df_enc_binary):

    # top_categories = 1
    encoder = OneHotEncoder(
        top_categories=1, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df_enc_binary)

    # test fit attr
    transf = {
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_C_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    transf = pd.DataFrame(transf).astype("int32")
    X = pd.DataFrame(X).astype("int32")

    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 4
    # test transform output
    pd.testing.assert_frame_equal(X, transf)
    assert "var_C_B" not in X.columns

    # top_categories = 2
    encoder = OneHotEncoder(
        top_categories=2, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df_enc_binary)

    # test fit attr
    transf = {
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_A_A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_B": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_C_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    transf = pd.DataFrame(transf).astype("int32")
    X = pd.DataFrame(X).astype("int32")

    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 4
    # test transform output
    pd.testing.assert_frame_equal(X, transf)
    assert "var_C_B" not in X.columns
