import re
from collections import Counter

import pandas as pd
import polars as pl
import pytest

from feature_engine.encoding import RareLabelEncoder
from tests.backend_helpers import to_dict

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer or set the parameter "
    "`missing_values='ignore'` when initialising this transformer."
)

FREQUENT_CATEGORIES = {
    "var_A": ["B", "D", "A", "G", "C"],
    "var_B": ["A", "D", "B", "G", "C"],
    "var_C": ["C", "D", "B", "G", "A"],
}

# data_enc_big after grouping categories E and F as "Rare"
ENC_BIG_RARE = {
    "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4 + ["D"] * 10 + ["Rare"] * 4 + ["G"] * 6,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4 + ["D"] * 10 + ["Rare"] * 4 + ["G"] * 6,
    "var_C": ["A"] * 4 + ["B"] * 6 + ["C"] * 10 + ["D"] * 10 + ["Rare"] * 4 + ["G"] * 6,
}


def test_defo_params_plus_automatically_find_variables(make_df, data_enc_big):
    # test case 1: defo params, automatically select variables
    encoder = RareLabelEncoder(
        tol=0.06, n_categories=5, variables=None, replace_with="Rare"
    )
    X = encoder.fit_transform(make_df(data_enc_big))

    # test init params
    assert encoder.tol == 0.06
    assert encoder.n_categories == 5
    assert encoder.replace_with == "Rare"
    assert encoder.variables is None
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 3
    assert encoder.encoder_dict_ == FREQUENT_CATEGORIES
    # test transform output
    assert isinstance(X, make_df)
    assert to_dict(X) == ENC_BIG_RARE


def test_when_varnames_are_numbers(data_enc_big):
    # integer column names are pandas-only, polars has no such concept
    input_df = pd.DataFrame(data_enc_big)
    input_df.columns = [1, 2, 3]

    encoder = RareLabelEncoder(
        tol=0.06, n_categories=5, variables=None, replace_with="Rare"
    )
    X = encoder.fit_transform(input_df)

    # expected output
    df = pd.DataFrame(
        {
            1: ENC_BIG_RARE["var_A"],
            2: ENC_BIG_RARE["var_B"],
            3: ENC_BIG_RARE["var_C"],
        }
    )

    assert encoder.variables_ == [1, 2, 3]
    assert encoder.encoder_dict_ == {
        1: FREQUENT_CATEGORIES["var_A"],
        2: FREQUENT_CATEGORIES["var_B"],
        3: FREQUENT_CATEGORIES["var_C"],
    }
    pd.testing.assert_frame_equal(X, df)


def test_correctly_ignores_nan_in_transform(make_df, data_enc_big):
    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=5,
        missing_values="ignore",
    )
    encoder.fit(make_df(data_enc_big))
    assert encoder.encoder_dict_ == FREQUENT_CATEGORIES

    X = encoder.transform(
        make_df(
            {
                "var_A": ["A", None, "J"],
                "var_B": ["A", None, "J"],
                "var_C": ["C", None, "J"],
            }
        )
    )

    assert isinstance(X, make_df)
    assert to_dict(X) == {
        "var_A": ["A", None, "Rare"],
        "var_B": ["A", None, "Rare"],
        "var_C": ["C", None, "Rare"],
    }


def test_correctly_ignores_nan_in_fit(make_df, data_enc_big):
    data = dict(data_enc_big)
    data["var_C"] = [None if v == "G" else v for v in data["var_C"]]

    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=3,
        missing_values="ignore",
    )
    encoder.fit(make_df(data))

    # expected:
    frequent_cat = {
        "var_A": ["B", "D", "A", "G", "C"],
        "var_B": ["A", "D", "B", "G", "C"],
        "var_C": ["C", "D", "B", "A"],
    }
    for key in frequent_cat.keys():
        assert Counter(encoder.encoder_dict_[key]) == Counter(frequent_cat[key])

    X = encoder.transform(
        make_df(
            {
                "var_A": ["A", None, "J", "G"],
                "var_B": ["A", None, "J", "G"],
                "var_C": ["C", None, "J", "G"],
            }
        )
    )

    assert isinstance(X, make_df)
    assert to_dict(X) == {
        "var_A": ["A", None, "Rare", "G"],
        "var_B": ["A", None, "Rare", "G"],
        "var_C": ["C", None, "Rare", "Rare"],
    }


def test_correctly_ignores_nan_in_fit_when_var_is_numerical(data_enc_big):
    # pandas .astype("O") mixed-dtype workaround for a numeric variable with
    # a string replace_with is a pandas-only quirk (polars casts to string
    # instead - see test_max_n_categories_with_numeric_var_polars).
    df = pd.DataFrame(data_enc_big)
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


def test_user_provides_grouping_label_name_and_variable_list(make_df, data_enc_big):
    # test case 2: user provides alternative grouping value and variable list
    encoder = RareLabelEncoder(
        tol=0.15, n_categories=5, variables=["var_A", "var_B"], replace_with="Other"
    )
    X = encoder.fit_transform(make_df(data_enc_big))

    # test init params
    assert encoder.tol == 0.15
    assert encoder.n_categories == 5
    assert encoder.replace_with == "Other"
    assert encoder.variables == ["var_A", "var_B"]
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.n_features_in_ == 3
    # test transform output
    assert isinstance(X, make_df)
    assert to_dict(X) == {
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
        "var_C": data_enc_big["var_C"],
    }


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


def test_warning_if_variable_cardinality_less_than_n_categories(
    make_df, data_enc_big
):
    # test case 3: when the variable has low cardinality
    msg = (
        "The number of unique categories for variable var_A is less than that "
        "indicated in n_categories. Thus, all categories will be "
        "considered frequent"
    )
    encoder = RareLabelEncoder(n_categories=10)
    with pytest.warns(UserWarning, match=re.escape(msg)):
        encoder.fit(make_df(data_enc_big))


def test_fit_raises_error_if_df_contains_na(make_df, data_enc_big_na):
    # test case 4: when dataset contains na, fit method
    encoder = RareLabelEncoder(n_categories=4)
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        encoder.fit(make_df(data_enc_big_na))


def test_transform_raises_error_if_df_contains_na(
    make_df, data_enc_big, data_enc_big_na
):
    # test case 5: when dataset contains na, transform method
    encoder = RareLabelEncoder(n_categories=4)
    encoder.fit(make_df(data_enc_big))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        encoder.transform(make_df(data_enc_big_na))


def test_max_n_categories(make_df, data_enc_big):
    # test case 6: user provides the maximum number of categories they want
    rare_encoder = RareLabelEncoder(tol=0.10, max_n_categories=4, n_categories=5)
    X = rare_encoder.fit_transform(make_df(data_enc_big))

    assert isinstance(X, make_df)
    assert to_dict(X) == {
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


def test_max_n_categories_with_numeric_var(data_enc_numeric):
    # pandas .astype("O") mixed-dtype workaround for a numeric variable with
    # a string replace_with is a pandas-only quirk (see the polars variant
    # below, which casts to string instead of keeping mixed dtypes).
    df_enc_numeric = pd.DataFrame(data_enc_numeric)
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


def test_max_n_categories_with_numeric_var_polars(data_enc_numeric):
    # polars can't hold mixed int/str values in one column like pandas'
    # object dtype does, so a numeric variable with a string replace_with
    # is cast to string entirely instead - a real, backend-specific
    # difference from the pandas behaviour above, not a bug.
    df_enc_numeric = pl.DataFrame(data_enc_numeric)
    rare_encoder = RareLabelEncoder(
        tol=0.10, max_n_categories=2, n_categories=1, ignore_format=True
    )

    X = rare_encoder.fit_transform(df_enc_numeric.select(["var_A", "var_B"]))

    assert isinstance(X, pl.DataFrame)
    assert to_dict(X) == {
        "var_A": ["1"] * 6 + ["2"] * 10 + ["Rare"] * 4,
        "var_B": ["1"] * 10 + ["2"] * 6 + ["Rare"] * 4,
    }


def test_inverse_transform_raises_not_implemented_error(make_df, data_enc_big):
    df_enc_big = make_df(data_enc_big)
    enc = RareLabelEncoder().fit(df_enc_big)
    with pytest.raises(NotImplementedError):
        enc.inverse_transform(df_enc_big)


def test_variables_cast_as_category(data_enc_big):
    # pandas category dtype is backend-specific: polars has no equivalent
    # concept in the same sense.
    df_enc_big = pd.DataFrame(data_enc_big)
    df_enc_big["var_B"] = df_enc_big["var_B"].astype("category")

    encoder = RareLabelEncoder(
        tol=0.06, n_categories=5, variables=None, replace_with="Rare"
    )
    X = encoder.fit_transform(df_enc_big)

    # expected output
    df = pd.DataFrame(ENC_BIG_RARE)
    df["var_B"] = pd.Categorical(df["var_B"])

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 3
    # test transform output
    pd.testing.assert_frame_equal(X, df, check_categorical=False)


def test_variables_cast_as_category_with_na_in_transform(data_enc_big):
    # pandas category dtype is backend-specific.
    df_enc_big = pd.DataFrame(data_enc_big)
    df_enc_big["var_B"] = df_enc_big["var_B"].astype("category")

    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=5,
        variables=None,
        replace_with="Rare",
        missing_values="ignore",
    )
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


def test_variables_cast_as_category_with_na_in_fit(data_enc_big):
    # pandas category dtype is backend-specific.
    df = pd.DataFrame(data_enc_big)
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
