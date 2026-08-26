from collections import Counter

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from feature_engine.encoding import RareLabelEncoder

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
DATA_ENC_BIG_NA = {
    key: [None] + values[1:] for key, values in DATA_ENC_BIG.items()
}
DATA_ENC_NUMERIC = {
    "var_A": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    "var_B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
}


def _to_pandas(X):
    return nw.from_native(X, eager_only=True).to_pandas()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_defo_params_plus_automatically_find_variables(make_df):
    # test case 1: defo params, automatically select variables
    df_enc_big = make_df(DATA_ENC_BIG)
    encoder = RareLabelEncoder(
        tol=0.06, n_categories=5, variables=None, replace_with="Rare"
    )
    X = encoder.fit_transform(df_enc_big)

    # expected output
    df = {
        "var_A": ["A"] * 6
        + ["B"] * 10
        + ["C"] * 4
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
        "var_B": ["A"] * 10
        + ["B"] * 6
        + ["C"] * 4
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
        "var_C": ["A"] * 4
        + ["B"] * 6
        + ["C"] * 10
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
    }
    df = pd.DataFrame(df)

    frequenc_cat = {
        "var_A": ["B", "D", "A", "G", "C"],
        "var_B": ["A", "D", "B", "G", "C"],
        "var_C": ["C", "D", "B", "G", "A"],
    }

    # test init params
    assert encoder.tol == 0.06
    assert encoder.n_categories == 5
    assert encoder.replace_with == "Rare"
    assert encoder.variables is None
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == frequenc_cat
    # test transform output
    pd.testing.assert_frame_equal(_to_pandas(X), df)


def test_when_varnames_are_numbers():
    # integer column names are pandas-only, polars has no such concept
    input_df = pd.DataFrame(DATA_ENC_BIG)
    input_df.columns = [1, 2, 3]

    encoder = RareLabelEncoder(
        tol=0.06, n_categories=5, variables=None, replace_with="Rare"
    )
    X = encoder.fit_transform(input_df)

    # expected output
    df = {
        1: ["A"] * 6 + ["B"] * 10 + ["C"] * 4 + ["D"] * 10 + ["Rare"] * 4 + ["G"] * 6,
        2: ["A"] * 10 + ["B"] * 6 + ["C"] * 4 + ["D"] * 10 + ["Rare"] * 4 + ["G"] * 6,
        3: ["A"] * 4 + ["B"] * 6 + ["C"] * 10 + ["D"] * 10 + ["Rare"] * 4 + ["G"] * 6,
    }
    df = pd.DataFrame(df)

    frequenc_cat = {
        1: ["B", "D", "A", "G", "C"],
        2: ["A", "D", "B", "G", "C"],
        3: ["C", "D", "B", "G", "A"],
    }

    assert encoder.variables_ == [1, 2, 3]
    assert encoder.encoder_dict_ == frequenc_cat
    pd.testing.assert_frame_equal(X, df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_correctly_ignores_nan_in_transform(make_df):
    df_enc_big = make_df(DATA_ENC_BIG)
    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=5,
        missing_values="ignore",
    )
    X = encoder.fit_transform(df_enc_big)

    # expected:
    frequenc_cat = {
        "var_A": ["B", "D", "A", "G", "C"],
        "var_B": ["A", "D", "B", "G", "C"],
        "var_C": ["C", "D", "B", "G", "A"],
    }
    assert encoder.encoder_dict_ == frequenc_cat

    # input
    t = make_df(
        {
            "var_A": ["A", None, "J"],
            "var_B": ["A", None, "J"],
            "var_C": ["C", None, "J"],
        }
    )

    # expected
    tt = pd.DataFrame(
        {
            "var_A": ["A", None, "Rare"],
            "var_B": ["A", None, "Rare"],
            "var_C": ["C", None, "Rare"],
        }
    )

    X = encoder.transform(t)
    pd.testing.assert_frame_equal(_to_pandas(X), tt)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_correctly_ignores_nan_in_fit(make_df):

    df = dict(DATA_ENC_BIG)
    df["var_C"] = [None if v == "G" else v for v in df["var_C"]]
    df = make_df(df)

    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=3,
        missing_values="ignore",
    )
    encoder.fit(df)

    # expected:
    frequent_cat = {
        "var_A": ["B", "D", "A", "G", "C"],
        "var_B": ["A", "D", "B", "G", "C"],
        "var_C": ["C", "D", "B", "A"],
    }
    for key in frequent_cat.keys():
        assert Counter(encoder.encoder_dict_[key]) == Counter(frequent_cat[key])

    # input
    t = make_df(
        {
            "var_A": ["A", None, "J", "G"],
            "var_B": ["A", None, "J", "G"],
            "var_C": ["C", None, "J", "G"],
        }
    )

    # expected
    tt = pd.DataFrame(
        {
            "var_A": ["A", None, "Rare", "G"],
            "var_B": ["A", None, "Rare", "G"],
            "var_C": ["C", None, "Rare", "Rare"],
        }
    )

    X = encoder.transform(t)
    pd.testing.assert_frame_equal(_to_pandas(X), tt)


def test_correctly_ignores_nan_in_fit_when_var_is_numerical():
    # pandas .astype("O") mixed-dtype workaround for a numeric variable with
    # a string replace_with is a pandas-only quirk (polars casts to string
    # instead - see test_max_n_categories_with_numeric_var_polars).
    df = pd.DataFrame(DATA_ENC_BIG)
    df["var_C"] = [
        1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6,
        None, None, None, None, None, None,
    ]

    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=3,
        missing_values="ignore",
        ignore_format=True,
    )
    encoder.fit(df)

    # expected:
    frequent_cat = {
        "var_A": ["B", "D", "A", "G", "C"],
        "var_B": ["A", "D", "B", "G", "C"],
        "var_C": [3, 4, 2, 1],
    }
    for key in frequent_cat.keys():
        assert Counter(encoder.encoder_dict_[key]) == Counter(frequent_cat[key])

    # input
    t = pd.DataFrame(
        {
            "var_A": ["A", None, "J", "G"],
            "var_B": ["A", None, "J", "G"],
            "var_C": [3, None, 9, 10],
        }
    )

    # expected (var_C mixes floats and strings after transform, so its
    # missing value must be an actual float nan, not a bare None, to match
    # pandas' own dtype inference for the same mix)
    tt = pd.DataFrame(
        {
            "var_A": ["A", None, "Rare", "G"],
            "var_B": ["A", None, "Rare", "G"],
            "var_C": [3.0, float("nan"), "Rare", "Rare"],
        }
    )

    X = encoder.transform(t)
    pd.testing.assert_frame_equal(X, tt, check_dtype=False)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_provides_grouping_label_name_and_variable_list(make_df):
    # test case 2: user provides alternative grouping value and variable list
    df_enc_big = make_df(DATA_ENC_BIG)
    encoder = RareLabelEncoder(
        tol=0.15, n_categories=5, variables=["var_A", "var_B"], replace_with="Other"
    )
    X = encoder.fit_transform(df_enc_big)

    # expected output
    df = {
        "var_A": ["A"] * 6
        + ["B"] * 10
        + ["Other"] * 4
        + ["D"] * 10
        + ["Other"] * 4
        + ["G"] * 6,
        "var_B": ["A"] * 10
        + ["B"] * 6
        + ["Other"] * 4
        + ["D"] * 10
        + ["Other"] * 4
        + ["G"] * 6,
        "var_C": ["A"] * 4
        + ["B"] * 6
        + ["C"] * 10
        + ["D"] * 10
        + ["E"] * 2
        + ["F"] * 2
        + ["G"] * 6,
    }
    df = pd.DataFrame(df)

    # test init params
    assert encoder.tol == 0.15
    assert encoder.n_categories == 5
    assert encoder.replace_with == "Other"
    assert encoder.variables == ["var_A", "var_B"]
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.n_features_in_ == 3
    # test transform output
    pd.testing.assert_frame_equal(_to_pandas(X), df)


# init params
@pytest.mark.parametrize("tol", ["hello", [0.5], -1, 1.5])
def test_error_if_tol_not_between_0_and_1(tol):
    with pytest.raises(ValueError):
        RareLabelEncoder(tol=tol)


@pytest.mark.parametrize("n_cat", ["hello", [0.5], -0.1, 1.5])
def test_error_if_n_categories_not_int(n_cat):
    with pytest.raises(ValueError):
        RareLabelEncoder(n_categories=n_cat)


@pytest.mark.parametrize("max_n_categories", ["hello", ["auto"], -1, 0.5])
def test_raises_error_when_max_n_categories_not_allowed(max_n_categories):
    with pytest.raises(ValueError):
        RareLabelEncoder(max_n_categories=max_n_categories)


@pytest.mark.parametrize("replace_with", [set("hello"), ["auto"]])
def test_error_if_replace_with_not_string(replace_with):
    with pytest.raises(ValueError):
        RareLabelEncoder(replace_with=replace_with)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_warning_if_variable_cardinality_less_than_n_categories(make_df):
    # test case 3: when the variable has low cardinality
    df_enc_big = make_df(DATA_ENC_BIG)
    with pytest.warns(UserWarning):
        encoder = RareLabelEncoder(n_categories=10)
        encoder.fit(df_enc_big)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_df_contains_na(make_df):
    # test case 4: when dataset contains na, fit method
    df_enc_big_na = make_df(DATA_ENC_BIG_NA)
    encoder = RareLabelEncoder(n_categories=4)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(df_enc_big_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_df_contains_na(make_df):
    # test case 5: when dataset contains na, transform method
    df_enc_big = make_df(DATA_ENC_BIG)
    df_enc_big_na = make_df(DATA_ENC_BIG_NA)
    encoder = RareLabelEncoder(n_categories=4)
    encoder.fit(df_enc_big)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.transform(df_enc_big_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_max_n_categories(make_df):
    # test case 6: user provides the maximum number of categories they want
    df_enc_big = make_df(DATA_ENC_BIG)
    rare_encoder = RareLabelEncoder(tol=0.10, max_n_categories=4, n_categories=5)
    X = rare_encoder.fit_transform(df_enc_big)
    df = {
        "var_A": ["A"] * 6
        + ["B"] * 10
        + ["Rare"] * 4
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
        "var_B": ["A"] * 10
        + ["B"] * 6
        + ["Rare"] * 4
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
        "var_C": ["Rare"] * 4
        + ["B"] * 6
        + ["C"] * 10
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
    }
    df = pd.DataFrame(df)
    pd.testing.assert_frame_equal(_to_pandas(X), df)


def test_max_n_categories_with_numeric_var():
    # pandas .astype("O") mixed-dtype workaround for a numeric variable with
    # a string replace_with is a pandas-only quirk (see the polars variant
    # below, which casts to string instead of keeping mixed dtypes).
    df_enc_numeric = pd.DataFrame(DATA_ENC_NUMERIC)
    rare_encoder = RareLabelEncoder(
        tol=0.10, max_n_categories=2, n_categories=1, ignore_format=True
    )

    X = rare_encoder.fit_transform(df_enc_numeric[["var_A", "var_B"]])

    df = df_enc_numeric[["var_A", "var_B"]].copy()
    df.replace({3: "Rare"}, inplace=True)

    # massive workaround because for some reason, doing a normal pd.assert_equal
    # was telling me that 2 columns that were identical, were actually not.
    # I think there was a problem with the type of each number perhaps
    for i in range(len(df)):
        assert str(list(X["var_A"])[i]) == str(list(df["var_A"])[i])
        assert str(list(X["var_B"])[i]) == str(list(df["var_B"])[i])


def test_max_n_categories_with_numeric_var_polars():
    # polars can't hold mixed int/str values in one column like pandas'
    # object dtype does, so a numeric variable with a string replace_with
    # is cast to string entirely instead - a real, backend-specific
    # difference from the pandas behaviour above, not a bug.
    df_enc_numeric = pl.DataFrame(DATA_ENC_NUMERIC)
    rare_encoder = RareLabelEncoder(
        tol=0.10, max_n_categories=2, n_categories=1, ignore_format=True
    )

    X = rare_encoder.fit_transform(df_enc_numeric.select(["var_A", "var_B"]))

    expected = pd.DataFrame(
        {
            "var_A": ["1"] * 6 + ["2"] * 10 + ["Rare"] * 4,
            "var_B": ["1"] * 10 + ["2"] * 6 + ["Rare"] * 4,
        }
    )
    pd.testing.assert_frame_equal(_to_pandas(X), expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_raises_not_implemented_error(make_df):
    df_enc_big = make_df(DATA_ENC_BIG)
    enc = RareLabelEncoder().fit(df_enc_big)
    with pytest.raises(NotImplementedError):
        enc.inverse_transform(df_enc_big)


def test_variables_cast_as_category():
    # pandas category dtype is backend-specific: polars has no equivalent
    # concept in the same sense.
    df_enc_big = pd.DataFrame(DATA_ENC_BIG)
    encoder = RareLabelEncoder(
        tol=0.06, n_categories=5, variables=None, replace_with="Rare"
    )

    df_enc_big = df_enc_big.copy()
    df_enc_big["var_B"] = df_enc_big["var_B"].astype("category")

    X = encoder.fit_transform(df_enc_big)

    # expected output
    df = {
        "var_A": ["A"] * 6
        + ["B"] * 10
        + ["C"] * 4
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
        "var_B": ["A"] * 10
        + ["B"] * 6
        + ["C"] * 4
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
        "var_C": ["A"] * 4
        + ["B"] * 6
        + ["C"] * 10
        + ["D"] * 10
        + ["Rare"] * 4
        + ["G"] * 6,
    }
    df = pd.DataFrame(df)
    df["var_B"] = pd.Categorical(df["var_B"])

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 3
    # test transform output
    pd.testing.assert_frame_equal(X, df, check_categorical=False)


def test_variables_cast_as_category_with_na_in_transform():
    df_enc_big = pd.DataFrame(DATA_ENC_BIG)
    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=5,
        variables=None,
        replace_with="Rare",
        missing_values="ignore",
    )

    df_enc_big = df_enc_big.copy()
    df_enc_big["var_B"] = df_enc_big["var_B"].astype("category")
    encoder.fit(df_enc_big)

    # input
    t = pd.DataFrame(
        {
            "var_A": ["A", None, "J", "G"],
            "var_B": ["A", None, "J", "G"],
            "var_C": ["A", None, "J", "G"],
        }
    )
    t["var_B"] = pd.Categorical(t["var_B"])

    # expected
    tt = pd.DataFrame(
        {
            "var_A": ["A", None, "Rare", "G"],
            "var_B": ["A", None, "Rare", "G"],
            "var_C": ["A", None, "Rare", "G"],
        }
    )
    tt["var_B"] = pd.Categorical(tt["var_B"])
    pd.testing.assert_frame_equal(encoder.transform(t), tt, check_categorical=False)


def test_variables_cast_as_category_with_na_in_fit():

    df = pd.DataFrame(DATA_ENC_BIG)
    df.loc[df["var_C"] == "G", "var_C"] = None
    df["var_C"] = df["var_C"].astype("category")

    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=3,
        missing_values="ignore",
    )
    encoder.fit(df)

    # input
    t = pd.DataFrame(
        {
            "var_A": ["A", None, "J", "G"],
            "var_B": ["A", None, "J", "G"],
            "var_C": ["C", None, "J", "G"],
        }
    )
    t["var_C"] = pd.Categorical(t["var_C"])

    # expected
    tt = pd.DataFrame(
        {
            "var_A": ["A", None, "Rare", "G"],
            "var_B": ["A", None, "Rare", "G"],
            "var_C": ["C", None, "Rare", "Rare"],
        }
    )
    tt["var_C"] = pd.Categorical(tt["var_C"])

    pd.testing.assert_frame_equal(encoder.transform(t), tt, check_categorical=False)
