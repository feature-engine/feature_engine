import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.preprocessing import MatchVariables

_params_fill_value = [
    (1, [1, 1, 1, 1], [1, 1, 1, 1]),
    (0.1, [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]),
    ("none", ["none", "none", "none", "none"], ["none", "none", "none", "none"]),
    (np.nan, [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]),
]

_params_allowed = [
    ([0, 1], "ignore", True),
    ("nan", "hola", True),
    ("nan", "ignore", "hallo"),
]


@pytest.mark.parametrize(
    "fill_value, expected_studies, expected_age", _params_fill_value
)
def test_drop_and_add_columns(
    fill_value, expected_studies, expected_age, df_vartypes, df_na
):
    train = df_na.copy()
    test = df_vartypes.copy()
    test = test.drop("Age", axis=1)  # to add more than one column

    # adding columns to test if they are removed
    for new_col in ["test1", "test2"]:
        test.loc[:, new_col] = new_col

    match_columns = MatchVariables(
        fill_value=fill_value,
        missing_values="ignore",
    )
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    expected_result = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Studies": expected_studies,
            "Age": expected_age,
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        }
    )

    # test init params
    if fill_value is np.nan:
        assert match_columns.fill_value is np.nan
    else:
        assert match_columns.fill_value == fill_value
    assert match_columns.verbose is True
    assert match_columns.missing_values == "ignore"
    # test fit attrs
    assert list(match_columns.feature_names_in_) == list(train.columns)
    assert match_columns.n_features_in_ == 6
    # test transform output
    pd.testing.assert_frame_equal(expected_result, transformed_df)


@pytest.mark.parametrize(
    "fill_value, expected_studies, expected_age", _params_fill_value
)
def test_columns_addition_when_more_columns_in_train_than_test(
    fill_value, expected_studies, expected_age, df_vartypes, df_na
):

    train = df_na.copy()
    test = df_vartypes.copy()
    test = test.drop("Age", axis=1)  # to add more than one column

    match_columns = MatchVariables(
        fill_value=fill_value,
        missing_values="ignore",
    )
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    expected_result = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Studies": expected_studies,
            "Age": expected_age,
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        }
    )

    # test init params
    if fill_value is np.nan:
        assert match_columns.fill_value is np.nan
    else:
        assert match_columns.fill_value == fill_value
    assert match_columns.verbose is True
    assert match_columns.missing_values == "ignore"
    # test fit attrs
    assert list(match_columns.feature_names_in_) == list(train.columns)
    assert match_columns.n_features_in_ == 6
    # test transform output
    pd.testing.assert_frame_equal(expected_result, transformed_df)


def test_drop_columns_when_more_columns_in_test_than_train(df_vartypes, df_na):
    train = df_vartypes.copy()
    train = train.drop("City", axis=1)  # to remove more than one column
    test = df_na.copy()

    match_columns = MatchVariables(missing_values="ignore")
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    expected_result = test.drop(columns=["Studies", "City"])

    # test init params
    assert match_columns.fill_value is np.nan
    assert match_columns.verbose is True
    assert match_columns.missing_values == "ignore"
    # test fit attrs
    assert list(match_columns.feature_names_in_) == list(train.columns)
    assert match_columns.n_features_in_ == 4
    # test transform output
    pd.testing.assert_frame_equal(expected_result, transformed_df)


@pytest.mark.parametrize("fill_value, missing_values, verbose", _params_allowed)
def test_error_if_param_values_not_allowed(fill_value, missing_values, verbose):
    with pytest.raises(ValueError):
        MatchVariables(
            fill_value=fill_value, missing_values=missing_values, verbose=verbose
        )


def test_verbose_print_out(capfd, df_vartypes, df_na):

    match_columns = MatchVariables(missing_values="ignore", verbose=True)

    train = df_na.copy()
    train.loc[:, "new_variable"] = 5

    match_columns.fit(train)
    match_columns.transform(df_vartypes)

    out, err = capfd.readouterr()
    assert (
        out == "The following variables are added to the DataFrame: "
        "['new_variable', 'Studies']\n"
        or out == "The following variables are added to the DataFrame: "
        "['Studies', 'new_variable']\n"
    )

    match_columns.fit(df_vartypes)
    match_columns.transform(train)

    out, err = capfd.readouterr()
    assert (
        out == "The following variables are dropped from the DataFrame: "
        "['new_variable', 'Studies']\n"
        or out == "The following variables are dropped from the DataFrame: "
        "['Studies', 'new_variable']\n"
    )


def test_raises_error_if_na_in_df(df_na, df_vartypes):
    # when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = MatchVariables()
        transformer.fit(df_na)

    # when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = MatchVariables()
        transformer.fit(df_vartypes)
        transformer.transform(df_na)


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = MatchVariables()
        transformer.transform(df_vartypes)
