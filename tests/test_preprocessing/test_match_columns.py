import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.preprocessing import MatchColumnsToTrainSet

_params_fill_value = [
    (1, [1, 1, 1, 1]),
    (0.1, [0.1, 0.1, 0.1, 0.1]),
    ("none", ["none", "none", "none", "none"]),
    (np.nan, [np.nan, np.nan, np.nan, np.nan]),
]

_params_allowed = [
    ([0, 1], "ignore", True),
    ("nan", "hola", True),
    ("nan", "ignore", "hallo"),
]


@pytest.mark.parametrize("fill_value, expected_studies", _params_fill_value)
def test_columns_addition_when_more_columns_in_train_than_test(
    fill_value, expected_studies, df_vartypes, df_na
):
    train = df_na.copy()
    test = df_vartypes.copy()

    match_columns = MatchColumnsToTrainSet(
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
            "Age": [20, 21, 19, 18],
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
    assert list(match_columns.input_features_) == list(train.columns)
    assert match_columns.n_features_in_ == 6
    # test transform output
    pd.testing.assert_frame_equal(expected_result, transformed_df)


def test_drop_columns_when_more_columns_in_test_than_train(df_vartypes, df_na):
    train = df_vartypes.copy()
    test = df_na.copy()

    match_columns = MatchColumnsToTrainSet(missing_values="ignore")
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    expected_result = test.drop(columns=["Studies"])

    # test init params
    assert match_columns.fill_value is np.nan
    assert match_columns.verbose is True
    assert match_columns.missing_values == "ignore"
    # test fit attrs
    assert list(match_columns.input_features_) == list(train.columns)
    assert match_columns.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(expected_result, transformed_df)


@pytest.mark.parametrize("fill_value, missing_values, verbose", _params_allowed)
def test_error_if_param_values_not_allowed(fill_value, missing_values, verbose):
    with pytest.raises(ValueError):
        MatchColumnsToTrainSet(
            fill_value=fill_value, missing_values=missing_values, verbose=verbose
        )


def test_verbose_print_out(capfd, df_vartypes, df_na):

    match_columns = MatchColumnsToTrainSet(missing_values="ignore")
    match_columns.fit(df_na)
    match_columns.transform(df_vartypes)

    out, err = capfd.readouterr()
    assert out == "The following variables are added to the DataFrame: ['Studies']\n"

    match_columns.fit(df_vartypes)
    match_columns.transform(df_na)

    out, err = capfd.readouterr()
    assert (
        out == "The following variables are dropped from the DataFrame: "
        "['Studies']\n"
    )


def test_raises_error_if_na_in_df(df_na, df_vartypes):
    # when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = MatchColumnsToTrainSet()
        transformer.fit(df_na)

    # when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = MatchColumnsToTrainSet()
        transformer.fit(df_vartypes)
        transformer.transform(df_na)


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = MatchColumnsToTrainSet()
        transformer.transform(df_vartypes)
