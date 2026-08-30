import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.encoding import OneHotEncoder

DF_ENC_BIG = {
    "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4 + ["D"] * 10 + ["E"] * 2
    + ["F"] * 2 + ["G"] * 6,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4 + ["D"] * 10 + ["E"] * 2
    + ["F"] * 2 + ["G"] * 6,
    "var_C": ["A"] * 4 + ["B"] * 6 + ["C"] * 10 + ["D"] * 10 + ["E"] * 2
    + ["F"] * 2 + ["G"] * 6,
}

DF_ENC_NUMERIC = {
    "var_A": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    "var_B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
}

DF_ENC_BINARY = {
    "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "var_C": ["AHA"] * 12 + ["UHU"] * 8,
    "var_D": ["OHO"] * 5 + ["EHE"] * 15,
    "var_num": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}


def _columns(X):
    return list(nw.from_native(X, eager_only=True).columns)


def _colsum(X, col):
    return sum(nw.from_native(X, eager_only=True).get_column(col).to_list())


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("index_", [[1, 2, 3], [3, 2, 1], [4, 9, 2]])
def test_concat_with_non_ordered_index(make_df, index_):
    data = {"varA": ["a", "b", "c"], "varB": ["d", "d", "a"]}
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
    result = nw.from_native(dft, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert list(result[col]) == values


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_categories_in_k_binary_plus_select_vars_automatically(make_df):
    # test case 1: encode all categories into k binary variables, select variables
    # automatically
    df = make_df(DF_ENC_BIG)
    encoder = OneHotEncoder(top_categories=None, variables=None, drop_last=False)
    X = encoder.fit_transform(df)

    # test init params
    assert encoder.top_categories is None
    assert encoder.variables is None
    assert encoder.drop_last is False
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
    for col, expected_sum in transf.items():
        assert _colsum(X, col) == expected_sum
    assert "var_A" not in _columns(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_categories_in_k_minus_1_binary_plus_list_of_variables(make_df):
    # test case 2: encode all categories into k-1 binary variables,
    # pass list of variables
    df = make_df(DF_ENC_BIG)
    encoder = OneHotEncoder(
        top_categories=None, variables=["var_A", "var_B"], drop_last=True
    )
    X = encoder.fit_transform(df)

    # test init params
    assert encoder.top_categories is None
    assert encoder.variables == ["var_A", "var_B"]
    assert encoder.drop_last is True
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
    columns = _columns(X)
    for col in transf.keys():
        assert _colsum(X, col) == transf[col]
    assert "var_B" not in columns
    assert "var_B_G" not in columns
    assert "var_C" in columns


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_top_categories(make_df):
    # test case 3: encode only the most popular categories
    data = {
        "var_A": ["A"] * 5 + ["B"] * 11 + ["C"] * 4 + ["D"] * 9 + ["E"] * 2
        + ["F"] * 2 + ["G"] * 7,
        "var_B": ["A"] * 11 + ["B"] * 7 + ["C"] * 4 + ["D"] * 9 + ["E"] * 2
        + ["F"] * 2 + ["G"] * 5,
        "var_C": ["A"] * 4 + ["B"] * 5 + ["C"] * 11 + ["D"] * 9 + ["E"] * 2
        + ["F"] * 2 + ["G"] * 7,
    }
    df = make_df(data)

    encoder = OneHotEncoder(top_categories=4, variables=None, drop_last=False)
    X = encoder.fit_transform(df)

    # test init params
    assert encoder.top_categories == 4
    # test fit attr
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
    columns = _columns(X)
    for col in transf.keys():
        assert _colsum(X, col) == transf[col]
    assert "var_B" not in columns
    assert "var_B_F" not in columns


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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_raises_error_if_df_contains_na(make_df):
    # test case 4: when dataset contains na, fit method
    data_na = dict(DF_ENC_BIG)
    data_na["var_A"] = [None] + list(DF_ENC_BIG["var_A"][1:])
    df_na = make_df(data_na)
    df = make_df(DF_ENC_BIG)

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )

    encoder = OneHotEncoder()
    with pytest.raises(ValueError, match=msg):
        encoder.fit(df_na)

    # test case 4: when dataset contains na, transform method
    encoder = OneHotEncoder()
    encoder.fit(df)
    with pytest.raises(ValueError, match=msg):
        encoder.transform(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_numerical_variables(make_df):
    df = make_df(DF_ENC_NUMERIC)
    encoder = OneHotEncoder(
        top_categories=None,
        variables=None,
        drop_last=False,
        ignore_format=True,
    )

    X = encoder.fit_transform(df)

    # test fit attr
    transf = {
        "var_A_1": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_A_2": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_A_3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "var_B_1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_B_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    }

    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.variables_binary_ == []
    assert encoder.n_features_in_ == 2
    assert encoder.encoder_dict_ == {"var_A": [1, 2, 3], "var_B": [1, 2, 3]}
    # test transform output
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    for col, values in transf.items():
        assert list(result[col]) == values


def test_variables_cast_as_category():
    # pandas-specific: category dtype has no polars equivalent behavior
    # under test here (encoding categorical-dtype columns).
    df = pd.DataFrame(DF_ENC_NUMERIC)
    df[["var_A", "var_B"]] = df[["var_A", "var_B"]].astype("category")

    encoder = OneHotEncoder(
        top_categories=None,
        variables=None,
        drop_last=False,
        ignore_format=True,
    )
    X = encoder.fit_transform(df)

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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_into_k_dummy_plus_drop_binary(make_df):
    df = make_df(DF_ENC_BINARY)
    encoder = OneHotEncoder(
        top_categories=None, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df)

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
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(transf.keys())
    for col, values in transf.items():
        assert list(result[col]) == values
    assert "var_C_B" not in result.keys()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_into_kminus1_dummyy_plus_drop_binary(make_df):
    df = make_df(DF_ENC_BINARY)
    encoder = OneHotEncoder(
        top_categories=None, variables=None, drop_last=True, drop_last_binary=True
    )
    X = encoder.fit_transform(df)

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
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(transf.keys())
    for col, values in transf.items():
        assert list(result[col]) == values
    assert "var_C_B" not in result.keys()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_into_top_categories_plus_drop_binary(make_df):
    df = make_df(DF_ENC_BINARY)
    # top_categories = 1
    encoder = OneHotEncoder(
        top_categories=1, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df)

    # test fit attr
    transf = {
        "var_num": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "var_A_B": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "var_B_A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_C_AHA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "var_D_OHO": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(transf.keys())
    for col, values in transf.items():
        assert list(result[col]) == values
    assert "var_C_B" not in result.keys()

    # top_categories = 2
    encoder = OneHotEncoder(
        top_categories=2, variables=None, drop_last=False, drop_last_binary=True
    )
    X = encoder.fit_transform(df)

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
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(transf.keys())
    for col, values in transf.items():
        assert list(result[col]) == values
    assert "var_C_B" not in result.keys()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out(make_df):
    df = make_df(DF_ENC_BINARY)
    original_features = ["var_num"]
    input_features = list(DF_ENC_BINARY.keys())

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

    with pytest.raises(ValueError):
        tr.get_feature_names_out("var_A")

    with pytest.raises(ValueError):
        tr.get_feature_names_out(["var_A", "hola"])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out_from_pipeline(make_df):
    df = make_df(DF_ENC_BINARY)
    original_features = ["var_num"]
    input_features = list(DF_ENC_BINARY.keys())

    tr = Pipeline([("transformer", OneHotEncoder())])
    tr.fit(df)

    out = [
        "var_A_A", "var_A_B", "var_A_C", "var_B_A", "var_B_B", "var_B_C",
        "var_C_AHA", "var_C_UHU", "var_D_OHO", "var_D_EHE",
    ]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=input_features) == feat_out


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_raises_not_implemented_error(make_df):
    df = make_df(DF_ENC_BINARY)
    enc = OneHotEncoder().fit(df)
    with pytest.raises(NotImplementedError):
        enc.inverse_transform(df)
