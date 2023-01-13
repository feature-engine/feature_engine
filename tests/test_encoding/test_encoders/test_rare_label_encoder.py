import numpy as np
import pandas as pd
import pytest

from feature_engine.encoding import RareLabelEncoder


def test_defo_params_plus_automatically_find_variables(df_enc_big):
    # test case 1: defo params, automatically select variables
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
    pd.testing.assert_frame_equal(X, df)


def test_correctly_ignores_nan_in_transform(df_enc_big):
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
    t = pd.DataFrame(
        {
            "var_A": ["A", np.nan, "J"],
            "var_B": ["A", np.nan, "J"],
            "var_C": ["C", np.nan, "J"],
        }
    )

    # expected
    tt = pd.DataFrame(
        {
            "var_A": ["A", np.nan, "Rare"],
            "var_B": ["A", np.nan, "Rare"],
            "var_C": ["C", np.nan, "Rare"],
        }
    )

    X = encoder.transform(t)
    pd.testing.assert_frame_equal(X, tt)


def test_correctly_ignores_nan_in_fit(df_enc_big):

    df = df_enc_big.copy()
    df.loc[df["var_C"] == "G", "var_C"] = np.nan

    encoder = RareLabelEncoder(
        tol=0.06,
        n_categories=3,
        missing_values="ignore",
    )
    X = encoder.fit_transform(df)

    # expected:
    frequenc_cat = {
        "var_A": ["B", "D", "A", "G", "C"],
        "var_B": ["A", "D", "B", "G", "C"],
        "var_C": ["C", "D", "B", "A"],
    }
    assert encoder.encoder_dict_ == frequenc_cat

    # input
    t = pd.DataFrame(
        {
            "var_A": ["A", np.nan, "J", "G"],
            "var_B": ["A", np.nan, "J", "G"],
            "var_C": ["C", np.nan, "J", "G"],
        }
    )

    # expected
    tt = pd.DataFrame(
        {
            "var_A": ["A", np.nan, "Rare", "G"],
            "var_B": ["A", np.nan, "Rare", "G"],
            "var_C": ["C", np.nan, "Rare", "Rare"],
        }
    )

    X = encoder.transform(t)
    pd.testing.assert_frame_equal(X, tt)


def test_user_provides_grouping_label_name_and_variable_list(df_enc_big):
    # test case 2: user provides alternative grouping value and variable list
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
    pd.testing.assert_frame_equal(X, df)


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


def test_warning_if_variable_cardinality_less_than_n_categories(df_enc_big):
    # test case 3: when the variable has low cardinality
    with pytest.warns(UserWarning):
        encoder = RareLabelEncoder(n_categories=10)
        encoder.fit(df_enc_big)


def test_fit_raises_error_if_df_contains_na(df_enc_big_na):
    # test case 4: when dataset contains na, fit method
    encoder = RareLabelEncoder(n_categories=4)
    with pytest.raises(ValueError) as record:
        msg = (
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer or set the parameter "
            "`missing_values='ignore'` when initialising this transformer."
        )
        encoder.fit(df_enc_big_na)
    assert str(record.value) == msg


def test_transform_raises_error_if_df_contains_na(df_enc_big, df_enc_big_na):
    # test case 5: when dataset contains na, transform method
    encoder = RareLabelEncoder(n_categories=4)
    encoder.fit(df_enc_big)
    with pytest.raises(ValueError) as record:
        msg = (
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer or set the parameter "
            "`missing_values='ignore'` when initialising this transformer."
        )
        encoder.transform(df_enc_big_na)
    assert str(record.value) == msg


def test_max_n_categories(df_enc_big):
    # test case 6: user provides the maximum number of categories they want
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
    pd.testing.assert_frame_equal(X, df)


def test_max_n_categories_with_numeric_var(df_enc_numeric):
    # ignore_format=True
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


def test_variables_cast_as_category(df_enc_big):
    # test case 1: defo params, automatically select variables
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

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B", "var_C"]
    assert encoder.n_features_in_ == 3
    # test transform output
    pd.testing.assert_frame_equal(X, df)


def test_inverse_transform_raises_not_implemented_error(df_enc_big):
    enc = RareLabelEncoder().fit(df_enc_big)
    with pytest.raises(NotImplementedError):
        enc.inverse_transform(df_enc_big)
