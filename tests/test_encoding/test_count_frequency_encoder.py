import warnings

import pandas as pd
import pytest
from numpy import nan
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import CountFrequencyEncoder


# init parameters
@pytest.mark.parametrize("enc_method", ["arbitrary", False, 1])
def test_error_if_encoding_method_not_permitted_value(enc_method):
    with pytest.raises(ValueError):
        CountFrequencyEncoder(encoding_method=enc_method)


@pytest.mark.parametrize(
    "errors", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_gets_not_permitted_value(errors):
    with pytest.raises(ValueError):
        CountFrequencyEncoder(unseen=errors)


@pytest.mark.parametrize(
    "params", [("count", "raise", True), ("frequency", "ignore", False)]
)
def test_init_param_assignment(params):
    enc = CountFrequencyEncoder(
        encoding_method=params[0],
        missing_values=params[1],
        ignore_format=params[2],
        unseen=params[1],
    )
    assert enc.encoding_method == params[0]
    assert enc.missing_values == params[1]
    assert enc.ignore_format == params[2]
    assert enc.unseen == params[1]


# fit and transform
def test_encode_1_variable_with_counts(df_enc):
    # test case 1: 1 variable, counts
    encoder = CountFrequencyEncoder(encoding_method="count", variables=["var_A"])
    X = encoder.fit_transform(df_enc)

    # expected result
    transf_df = df_enc.copy()
    transf_df["var_A"] = [
        6,
        6,
        6,
        6,
        6,
        6,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        4,
        4,
        4,
        4,
    ]

    # init params
    assert encoder.encoding_method == "count"
    assert encoder.variables == ["var_A"]
    # fit params
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {"A": 6, "B": 10, "C": 4}}
    assert encoder.n_features_in_ == 3
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)


def test_automatically_select_variables_encode_with_frequency(df_enc):
    # test case 2: automatically select variables, frequency
    encoder = CountFrequencyEncoder(encoding_method="frequency", variables=None)
    X = encoder.fit_transform(df_enc)

    # expected output
    transf_df = df_enc.copy()
    transf_df["var_A"] = [
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.2,
        0.2,
        0.2,
        0.2,
    ]
    transf_df["var_B"] = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.2,
        0.2,
        0.2,
        0.2,
    ]

    # init params
    assert encoder.encoding_method == "frequency"
    assert encoder.variables is None
    # fit params
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.3, "B": 0.5, "C": 0.2},
        "var_B": {"A": 0.5, "B": 0.3, "C": 0.2},
    }
    assert encoder.n_features_in_ == 3
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)


def test_encoding_when_nan_in_fit_df(df_enc):
    df = df_enc.copy()
    df.loc[len(df)] = [nan, nan, nan]

    encoder = CountFrequencyEncoder(
        encoding_method="frequency",
        missing_values="ignore",
    )
    encoder.fit(df_enc)

    X = encoder.transform(
        pd.DataFrame({"var_A": ["A", nan], "var_B": ["A", nan], "target": [1, 0]})
    )

    # transform params
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"var_A": [0.3, nan], "var_B": [0.5, nan], "target": [1, 0]}),
    )


@pytest.mark.parametrize("enc_method", ["arbitrary", False, 1])
def test_error_if_encoding_method_not_recognized_in_fit(enc_method, df_enc):
    enc = CountFrequencyEncoder()
    enc.encoding_method = enc_method
    with pytest.raises(ValueError) as record:
        enc.fit(df_enc)
    msg = (
        "Unrecognized value for encoding_method. It should be 'count' or "
        f"'frequency'. Got {enc_method} instead."
    )
    assert str(record.value) == msg


def test_warning_when_df_contains_unseen_categories(df_enc, df_enc_rare):
    # dataset to be transformed contains categories not present in
    # training dataset (unseen categories), unseen set to ignore.

    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when unseen equals 'ignore'
    encoder = CountFrequencyEncoder(unseen="ignore")
    encoder.fit(df_enc)
    with pytest.warns(UserWarning) as record:
        encoder.transform(df_enc_rare)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg


def test_error_when_df_contains_unseen_categories(df_enc, df_enc_rare):
    # dataset to be transformed contains categories not present in
    # training dataset (unseen categories), unseen set to raise.

    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    encoder = CountFrequencyEncoder(unseen="raise")
    encoder.fit(df_enc)

    # check for exception when unseen equals 'raise'
    with pytest.raises(ValueError) as record:
        encoder.transform(df_enc_rare)

    # check that the error message matches
    assert str(record.value) == msg

    # check for no error and no warning when unseen equals 'encode'
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encoder = CountFrequencyEncoder(unseen="encode")
        encoder.fit(df_enc)
        encoder.transform(df_enc_rare)


def test_no_error_triggered_when_df_contains_unseen_categories_and_unseen_is_encode(
    df_enc, df_enc_rare
):
    # dataset to be transformed contains categories not present in
    # training dataset (unseen categories).

    # check for no error and no warning when unseen equals 'encode'
    warnings.simplefilter("error")
    encoder = CountFrequencyEncoder(unseen="encode")
    encoder.fit(df_enc)
    with warnings.catch_warnings():
        encoder.transform(df_enc_rare)


@pytest.mark.parametrize("errors", ["raise", "ignore", "encode"])
def test_fit_raises_error_if_df_contains_na(errors, df_enc_na):
    # test case 4: when dataset contains na, fit method
    encoder = CountFrequencyEncoder(unseen=errors)
    with pytest.raises(ValueError) as record:
        encoder.fit(df_enc_na)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


@pytest.mark.parametrize("errors", ["raise", "ignore", "encode"])
def test_transform_raises_error_if_df_contains_na(errors, df_enc, df_enc_na):
    # test case 4: when dataset contains na, transform method
    encoder = CountFrequencyEncoder(unseen=errors)
    encoder.fit(df_enc)
    with pytest.raises(ValueError) as record:
        encoder.transform(df_enc_na)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_zero_encoding_for_new_categories():

    df_fit = pd.DataFrame(
        {"col1": ["a", "a", "b", "a", "c"], "col2": ["1", "2", "3", "1", "2"]}
    )
    df_transf = pd.DataFrame(
        {"col1": ["a", "d", "b", "a", "c"], "col2": ["1", "2", "3", "1", "4"]}
    )
    encoder = CountFrequencyEncoder(unseen="encode").fit(df_fit)

    result = encoder.transform(df_transf)

    # check that no NaNs are added
    assert pd.isnull(result).sum().sum() == 0

    # check that the counts are correct for both new and old
    expected_result = pd.DataFrame({"col1": [3, 0, 1, 3, 1], "col2": [2, 2, 1, 2, 0]})
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


def test_zero_encoding_for_unseen_categories_if_unseen_is_encode():
    df_fit = pd.DataFrame(
        {"col1": ["a", "a", "b", "a", "c"], "col2": ["1", "2", "3", "1", "2"]}
    )
    df_transform = pd.DataFrame(
        {"col1": ["a", "d", "b", "a", "c"], "col2": ["1", "2", "3", "1", "4"]}
    )

    # count encoding
    encoder = CountFrequencyEncoder(unseen="encode").fit(df_fit)
    result = encoder.transform(df_transform)

    # check that no NaNs are added
    assert pd.isnull(result).sum().sum() == 0

    # check that the counts are correct
    expected_result = pd.DataFrame({"col1": [3, 0, 1, 3, 1], "col2": [2, 2, 1, 2, 0]})
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)

    # with frequency
    encoder = CountFrequencyEncoder(encoding_method="frequency", unseen="encode").fit(
        df_fit
    )
    result = encoder.transform(df_transform)

    # check that no NaNs are added
    assert pd.isnull(result).sum().sum() == 0

    # check that the frequencies are correct
    expected_result = pd.DataFrame(
        {"col1": [0.6, 0, 0.2, 0.6, 0.2], "col2": [0.4, 0.4, 0.2, 0.4, 0]}
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_nan_encoding_for_new_categories_if_unseen_is_ignore():
    df_fit = pd.DataFrame(
        {"col1": ["a", "a", "b", "a", "c"], "col2": ["1", "2", "3", "1", "2"]}
    )
    df_transf = pd.DataFrame(
        {"col1": ["a", "d", "b", "a", "c"], "col2": ["1", "2", "3", "1", "4"]}
    )
    encoder = CountFrequencyEncoder(unseen="ignore").fit(df_fit)
    result = encoder.transform(df_transf)

    # check that no NaNs are added
    assert pd.isnull(result).sum().sum() == 2

    # check that the counts are correct for both new and old
    expected_result = pd.DataFrame(
        {"col1": [3, nan, 1, 3, 1], "col2": [2, 2, 1, 2, nan]}
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_ignore_variable_format_with_frequency(df_vartypes):
    encoder = CountFrequencyEncoder(
        encoding_method="frequency", variables=None, ignore_format=True
    )
    X = encoder.fit_transform(df_vartypes)

    # expected output
    transf_df = {
        "Name": [0.25, 0.25, 0.25, 0.25],
        "City": [0.25, 0.25, 0.25, 0.25],
        "Age": [0.25, 0.25, 0.25, 0.25],
        "Marks": [0.25, 0.25, 0.25, 0.25],
        "dob": [0.25, 0.25, 0.25, 0.25],
    }

    transf_df = pd.DataFrame(transf_df)

    # init params
    assert encoder.encoding_method == "frequency"
    assert encoder.variables is None
    # fit params
    assert encoder.variables_ == ["Name", "City", "Age", "Marks", "dob"]
    assert encoder.n_features_in_ == 5
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)


def test_column_names_are_numbers(df_numeric_columns):
    encoder = CountFrequencyEncoder(
        encoding_method="frequency", variables=[0, 1, 2, 3], ignore_format=True
    )
    X = encoder.fit_transform(df_numeric_columns)

    # expected output
    transf_df = {
        0: [0.25, 0.25, 0.25, 0.25],
        1: [0.25, 0.25, 0.25, 0.25],
        2: [0.25, 0.25, 0.25, 0.25],
        3: [0.25, 0.25, 0.25, 0.25],
        4: pd.date_range("2020-02-24", periods=4, freq="min"),
    }

    transf_df = pd.DataFrame(transf_df)

    # init params
    assert encoder.encoding_method == "frequency"
    assert encoder.variables == [0, 1, 2, 3]
    # fit params
    assert encoder.variables_ == [0, 1, 2, 3]
    assert encoder.n_features_in_ == 5
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)


def test_variables_cast_as_category(df_enc_category_dtypes):
    encoder = CountFrequencyEncoder(encoding_method="count", variables=["var_A"])
    X = encoder.fit_transform(df_enc_category_dtypes)

    # expected result
    transf_df = df_enc_category_dtypes.copy()
    transf_df["var_A"] = [
        6,
        6,
        6,
        6,
        6,
        6,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        4,
        4,
        4,
        4,
    ]
    # transform params
    pd.testing.assert_frame_equal(X, transf_df, check_dtype=False)
    assert X["var_A"].dtypes == int

    encoder = CountFrequencyEncoder(encoding_method="frequency", variables=["var_A"])
    X = encoder.fit_transform(df_enc_category_dtypes)
    assert X["var_A"].dtypes == float


def test_inverse_transform_when_no_unseen():
    df = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = CountFrequencyEncoder()
    enc.fit(df)
    dft = enc.transform(df)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df)


def test_inverse_transform_when_ignore_unseen():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    df3 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", nan]})
    enc = CountFrequencyEncoder(unseen="ignore")
    enc.fit(df1)
    dft = enc.transform(df2)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df3)


def test_inverse_transform_when_encode_unseen():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    df3 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", nan]})
    enc = CountFrequencyEncoder(unseen="encode")
    enc.fit(df1)
    dft = enc.transform(df2)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df3)


def test_inverse_transform_raises_non_fitted_error():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = CountFrequencyEncoder()

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)

    df1.loc[len(df1) - 1] = nan

    with pytest.raises(ValueError):
        enc.fit(df1)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)
