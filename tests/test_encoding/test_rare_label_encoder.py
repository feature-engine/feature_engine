import pytest
import pandas as pd

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

    # test init params
    assert encoder.tol == 0.06
    assert encoder.n_categories == 5
    assert encoder.replace_with == "Rare"
    assert encoder.variables == ["var_A", "var_B", "var_C"]
    # test fit attr
    assert encoder.input_shape_ == (40, 3)
    # test transform output
    pd.testing.assert_frame_equal(X, df)


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
    assert encoder.input_shape_ == (40, 3)
    # test transform output
    pd.testing.assert_frame_equal(X, df)


def test_error_if_tol_not_between_0_and_1():
    with pytest.raises(ValueError):
        RareLabelEncoder(tol=5)


def test_error_if_n_categories_not_int():
    with pytest.raises(ValueError):
        RareLabelEncoder(n_categories=0.5)


def test_error_if_replace_with_not_string():
    with pytest.raises(ValueError):
        RareLabelEncoder(replace_with=0.5)


def test_warning_if_variable_cardinality_less_than_n_categories(df_enc_big):
    # test case 3: when the variable has low cardinality
    with pytest.warns(UserWarning):
        encoder = RareLabelEncoder(n_categories=10)
        encoder.fit(df_enc_big)


def test_fit_raises_error_if_df_contains_na(df_enc_big_na):
    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = RareLabelEncoder(n_categories=4)
        encoder.fit(df_enc_big_na)


def test_transform_raises_error_if_df_contains_na(df_enc_big, df_enc_big_na):
    # test case 5: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = RareLabelEncoder(n_categories=4)
        encoder.fit(df_enc_big)
        encoder.transform(df_enc_big_na)


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
