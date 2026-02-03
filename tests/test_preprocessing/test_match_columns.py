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
    ([0, 1], "ignore", True, True),
    ("nan", "hola", True, True),
    ("nan", "ignore", True, "hallo"),
    ("nan", "ignore", "hallo", True),
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
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
        }
    )

    # test init params
    if fill_value is np.nan:
        assert match_columns.fill_value is np.nan
    else:
        assert match_columns.fill_value == fill_value
    assert match_columns.verbose is True
    assert match_columns.missing_values == "ignore"
    assert match_columns.match_dtypes is False
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
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
        }
    )

    # test init params
    if fill_value is np.nan:
        assert match_columns.fill_value is np.nan
    else:
        assert match_columns.fill_value == fill_value
    assert match_columns.verbose is True
    assert match_columns.missing_values == "ignore"
    assert match_columns.match_dtypes is False
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
    assert match_columns.match_dtypes is False
    # test fit attrs
    assert list(match_columns.feature_names_in_) == list(train.columns)
    assert match_columns.n_features_in_ == 4
    # test transform output
    pd.testing.assert_frame_equal(expected_result, transformed_df)


def test_match_dtypes_string_to_numbers(df_vartypes):
    train = df_vartypes.copy().select_dtypes("number")
    test = train.copy().astype("string")

    match_columns = MatchVariables(match_dtypes=True)
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    # test init params
    assert match_columns.match_dtypes is True
    # test fit attrs
    assert match_columns.dtype_dict_ == {
        "Age": np.dtype("int64"),
        "Marks": np.dtype("float64"),
    }

    # test transform output
    pd.testing.assert_series_equal(train.dtypes, transformed_df.dtypes)
    pd.testing.assert_frame_equal(transformed_df, train)


def test_match_dtypes_numbers_to_string(df_vartypes):
    train = df_vartypes.copy().select_dtypes("number").astype("string")
    test = df_vartypes.copy().select_dtypes("number")

    match_columns = MatchVariables(match_dtypes=True)
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    # test init params
    assert match_columns.match_dtypes is True
    # test fit attrs
    assert isinstance(match_columns.dtype_dict_, dict)
    # test transform output
    pd.testing.assert_series_equal(train.dtypes, transformed_df.dtypes)
    pd.testing.assert_frame_equal(transformed_df, train)


def test_match_dtypes_string_to_datetime(df_vartypes):
    train = df_vartypes.copy().loc[:, ["dob"]]
    test = train.copy().astype("string")

    match_columns = MatchVariables(match_dtypes=True, verbose=False)
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    # test init params
    assert match_columns.match_dtypes is True
    assert match_columns.verbose is False
    # test fit attrs
    assert match_columns.dtype_dict_ == {"dob": np.dtype("<M8[ns]")}
    # test transform output
    pd.testing.assert_series_equal(train.dtypes, transformed_df.dtypes)
    pd.testing.assert_frame_equal(transformed_df, train)


def test_match_dtypes_datetime_to_string(df_vartypes):
    train = df_vartypes.copy().loc[:, ["dob"]].astype("string")
    test = df_vartypes.copy().loc[:, ["dob"]]

    match_columns = MatchVariables(match_dtypes=True, verbose=False)
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    # test init params
    assert match_columns.match_dtypes is True
    assert match_columns.verbose is False
    # test fit attrs
    assert isinstance(match_columns.dtype_dict_, dict)
    # test transform output
    pd.testing.assert_series_equal(train.dtypes, transformed_df.dtypes)
    pd.testing.assert_frame_equal(transformed_df, train)


def test_match_dtypes_missing_category(df_vartypes):
    train = df_vartypes.copy().loc[:, ["Name", "City"]].astype("category")
    test = df_vartypes.copy().loc[:, ["Name", "City"]].iloc[:-1].astype("category")

    match_columns = MatchVariables(match_dtypes=True, verbose=True)
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    # test init params
    assert match_columns.match_dtypes is True
    assert match_columns.verbose is True
    # test fit attrs
    assert match_columns.dtype_dict_ == {
        "Name": pd.CategoricalDtype(
            categories=["jack", "krish", "nick", "tom"], ordered=False
        ),
        "City": pd.CategoricalDtype(
            categories=["Bristol", "Liverpool", "London", "Manchester"], ordered=False
        ),
    }
    # test transform output
    pd.testing.assert_series_equal(train.dtypes, transformed_df.dtypes)
    pd.testing.assert_frame_equal(transformed_df, train.iloc[:-1])


def test_match_dtypes_extra_category(df_vartypes):
    train = df_vartypes.copy().loc[:, ["Name", "City"]].iloc[:-1].astype("category")
    test = df_vartypes.copy().loc[:, ["Name", "City"]].astype("category")

    match_columns = MatchVariables(match_dtypes=True, verbose=True)
    match_columns.fit(train)

    transformed_df = match_columns.transform(test)

    # test init params
    assert match_columns.match_dtypes is True
    assert match_columns.verbose is True
    # test fit attrs
    assert match_columns.dtype_dict_ == {
        "Name": pd.CategoricalDtype(categories=["krish", "nick", "tom"], ordered=False),
        "City": pd.CategoricalDtype(
            categories=["Liverpool", "London", "Manchester"], ordered=False
        ),
    }

    # test transform output
    pd.testing.assert_series_equal(train.dtypes, transformed_df.dtypes)


@pytest.mark.parametrize(
    "fill_value, missing_values, match_dtypes, verbose", _params_allowed
)
def test_error_if_param_values_not_allowed(
    fill_value, missing_values, match_dtypes, verbose
):
    with pytest.raises(ValueError):
        MatchVariables(
            fill_value=fill_value,
            missing_values=missing_values,
            match_dtypes=match_dtypes,
            verbose=verbose,
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


def test_check_for_na_in_transform_does_not_fail():
    # some variables seen during train may not be present in the test set, so we
    # need to skip these ones during the check for na.
    # Bug reported in https://github.com/feature-engine/feature_engine/issues/789
    train = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
        }
    )

    test = pd.DataFrame(
        {
            "Name": ["tom", "sam", "nick"],
            "Age": [20, 22, 23],
            "Marks": [0.9, 0.7, 0.6],
            "Hobbies": ["tennis", "rugby", "football"],
        }
    )

    expected = pd.DataFrame(
        {
            "Name": ["tom", "sam", "nick"],
            "City": [np.nan, np.nan, np.nan],
            "Age": [20, 22, 23],
            "Marks": [0.9, 0.7, 0.6],
        }
    )

    match_columns = MatchVariables()
    match_columns.fit(train)
    df_transformed = match_columns.transform(test)

    assert isinstance(df_transformed, pd.DataFrame)
    pd.testing.assert_frame_equal(df_transformed, expected)
