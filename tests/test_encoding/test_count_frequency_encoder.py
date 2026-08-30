import re
import warnings

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import CountEncoder, CountFrequencyEncoder

DATA_ENC = {
    "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}
DATA_ENC_RARE = {
    "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}
DATA_ENC_NA = {
    "var_A": [None] + ["B"] * 8 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}
DATA_VARTYPES = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
    "dob": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
}


def _to_pandas(X):
    return nw.from_native(X, eager_only=True).to_pandas()


def _null_count(X):
    nw_X = nw.from_native(X, eager_only=True)
    return sum(nw_X.get_column(c).null_count() for c in nw_X.columns)


# init parameters
@pytest.mark.parametrize("enc_method", ["arbitrary", False, 1])
def test_error_if_encoding_method_not_permitted_value(enc_method):
    with pytest.raises(ValueError):
        CountEncoder(encoding_method=enc_method)


@pytest.mark.parametrize(
    "errors", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_gets_not_permitted_value(errors):
    with pytest.raises(ValueError):
        CountEncoder(unseen=errors)


@pytest.mark.parametrize(
    "params", [("count", "raise", True), ("frequency", "ignore", False)]
)
def test_init_param_assignment(params):
    enc = CountEncoder(
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
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encode_1_variable_with_counts(make_df):
    # test case 1: 1 variable, counts
    df_enc = make_df(DATA_ENC)
    encoder = CountEncoder(encoding_method="count", variables=["var_A"])
    X = encoder.fit_transform(df_enc)

    # expected result
    transf_df = _to_pandas(df_enc)
    transf_df["var_A"] = [6] * 6 + [10] * 10 + [4] * 4

    # init params
    assert encoder.encoding_method == "count"
    assert encoder.variables == ["var_A"]
    # fit params
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {"A": 6, "B": 10, "C": 4}}
    assert encoder.n_features_in_ == 3
    # transform params
    pd.testing.assert_frame_equal(_to_pandas(X), transf_df, check_dtype=False)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_select_variables_encode_with_frequency(make_df):
    # test case 2: automatically select variables, frequency
    df_enc = make_df(DATA_ENC)
    encoder = CountEncoder(encoding_method="frequency", variables=None)
    X = encoder.fit_transform(df_enc)

    # expected output
    transf_df = _to_pandas(df_enc)
    transf_df["var_A"] = [0.3] * 6 + [0.5] * 10 + [0.2] * 4
    transf_df["var_B"] = [0.5] * 10 + [0.3] * 6 + [0.2] * 4

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
    pd.testing.assert_frame_equal(_to_pandas(X), transf_df, check_dtype=False)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encoding_when_nan_in_fit_df(make_df):
    df_enc = make_df(DATA_ENC)

    encoder = CountEncoder(
        encoding_method="frequency",
        missing_values="ignore",
    )
    encoder.fit(df_enc)

    X = encoder.transform(
        make_df({"var_A": ["A", None], "var_B": ["A", None], "target": [1, 0]})
    )

    # transform params
    result = _to_pandas(X)
    expected = pd.DataFrame(
        {"var_A": [0.3, None], "var_B": [0.5, None], "target": [1, 0]}
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("enc_method", ["arbitrary", False, 1])
def test_error_if_encoding_method_not_recognized_in_fit(enc_method, make_df):
    df_enc = make_df(DATA_ENC)
    enc = CountEncoder()
    enc.encoding_method = enc_method
    msg = (
        "Unrecognized value for encoding_method. It should be 'count' or "
        f"'frequency'. Got {enc_method} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        enc.fit(df_enc)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_warning_when_df_contains_unseen_categories(make_df):
    # dataset to be transformed contains categories not present in
    # training dataset (unseen categories), unseen set to ignore.
    df_enc = make_df(DATA_ENC)
    df_enc_rare = make_df(DATA_ENC_RARE)

    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when unseen equals 'ignore'
    encoder = CountEncoder(unseen="ignore")
    encoder.fit(df_enc)
    with pytest.warns(UserWarning) as record:
        encoder.transform(df_enc_rare)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_df_contains_unseen_categories(make_df):
    # dataset to be transformed contains categories not present in
    # training dataset (unseen categories), unseen set to raise.
    df_enc = make_df(DATA_ENC)
    df_enc_rare = make_df(DATA_ENC_RARE)

    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    encoder = CountEncoder(unseen="raise")
    encoder.fit(df_enc)

    # check for exception when unseen equals 'raise'
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(df_enc_rare)

    # check for no error and no warning when unseen equals 'encode'
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encoder = CountEncoder(unseen="encode")
        encoder.fit(df_enc)
        encoder.transform(df_enc_rare)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_no_error_triggered_when_df_contains_unseen_categories_and_unseen_is_encode(
    make_df,
):
    # dataset to be transformed contains categories not present in
    # training dataset (unseen categories).
    df_enc = make_df(DATA_ENC)
    df_enc_rare = make_df(DATA_ENC_RARE)

    # check for no error and no warning when unseen equals 'encode'
    warnings.simplefilter("error")
    encoder = CountEncoder(unseen="encode")
    encoder.fit(df_enc)
    with warnings.catch_warnings():
        encoder.transform(df_enc_rare)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("errors", ["raise", "ignore", "encode"])
def test_fit_raises_error_if_df_contains_na(errors, make_df):
    # test case 4: when dataset contains na, fit method
    df_enc_na = make_df(DATA_ENC_NA)
    encoder = CountEncoder(unseen=errors)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.fit(df_enc_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("errors", ["raise", "ignore", "encode"])
def test_transform_raises_error_if_df_contains_na(errors, make_df):
    # test case 4: when dataset contains na, transform method
    df_enc = make_df(DATA_ENC)
    df_enc_na = make_df(DATA_ENC_NA)
    encoder = CountEncoder(unseen=errors)
    encoder.fit(df_enc)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(df_enc_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_zero_encoding_for_new_categories(make_df):
    df_fit = make_df(
        {"col1": ["a", "a", "b", "a", "c"], "col2": ["1", "2", "3", "1", "2"]}
    )
    df_transf = make_df(
        {"col1": ["a", "d", "b", "a", "c"], "col2": ["1", "2", "3", "1", "4"]}
    )
    encoder = CountEncoder(unseen="encode").fit(df_fit)

    result = encoder.transform(df_transf)

    # check that no NaNs are added
    assert _null_count(result) == 0

    # check that the counts are correct for both new and old
    expected_result = pd.DataFrame({"col1": [3, 0, 1, 3, 1], "col2": [2, 2, 1, 2, 0]})
    pd.testing.assert_frame_equal(
        _to_pandas(result), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_zero_encoding_for_unseen_categories_if_unseen_is_encode(make_df):
    df_fit = make_df(
        {"col1": ["a", "a", "b", "a", "c"], "col2": ["1", "2", "3", "1", "2"]}
    )
    df_transform = make_df(
        {"col1": ["a", "d", "b", "a", "c"], "col2": ["1", "2", "3", "1", "4"]}
    )

    # count encoding
    encoder = CountEncoder(unseen="encode").fit(df_fit)
    result = encoder.transform(df_transform)

    # check that no NaNs are added
    assert _null_count(result) == 0

    # check that the counts are correct
    expected_result = pd.DataFrame({"col1": [3, 0, 1, 3, 1], "col2": [2, 2, 1, 2, 0]})
    pd.testing.assert_frame_equal(
        _to_pandas(result), expected_result, check_dtype=False
    )

    # with frequency
    encoder = CountEncoder(encoding_method="frequency", unseen="encode").fit(df_fit)
    result = encoder.transform(df_transform)

    # check that no NaNs are added
    assert _null_count(result) == 0

    # check that the frequencies are correct
    expected_result = pd.DataFrame(
        {"col1": [0.6, 0, 0.2, 0.6, 0.2], "col2": [0.4, 0.4, 0.2, 0.4, 0]}
    )
    pd.testing.assert_frame_equal(
        _to_pandas(result), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_nan_encoding_for_new_categories_if_unseen_is_ignore(make_df):
    df_fit = make_df(
        {"col1": ["a", "a", "b", "a", "c"], "col2": ["1", "2", "3", "1", "2"]}
    )
    df_transf = make_df(
        {"col1": ["a", "d", "b", "a", "c"], "col2": ["1", "2", "3", "1", "4"]}
    )
    encoder = CountEncoder(unseen="ignore").fit(df_fit)
    result = encoder.transform(df_transf)

    # check that 2 NaNs are added
    assert _null_count(result) == 2

    # check that the counts are correct for both new and old
    expected_result = pd.DataFrame(
        {"col1": [3, None, 1, 3, 1], "col2": [2, 2, 1, 2, None]}
    )
    pd.testing.assert_frame_equal(
        _to_pandas(result), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_ignore_variable_format_with_frequency(make_df):
    df_vartypes = make_df(DATA_VARTYPES)
    encoder = CountEncoder(
        encoding_method="frequency", variables=None, ignore_format=True
    )
    X = encoder.fit_transform(df_vartypes)

    # expected output
    transf_df = pd.DataFrame(
        {
            "Name": [0.25, 0.25, 0.25, 0.25],
            "City": [0.25, 0.25, 0.25, 0.25],
            "Age": [0.25, 0.25, 0.25, 0.25],
            "Marks": [0.25, 0.25, 0.25, 0.25],
            "dob": [0.25, 0.25, 0.25, 0.25],
        }
    )

    # init params
    assert encoder.encoding_method == "frequency"
    assert encoder.variables is None
    # fit params
    assert encoder.variables_ == ["Name", "City", "Age", "Marks", "dob"]
    assert encoder.n_features_in_ == 5
    # transform params
    pd.testing.assert_frame_equal(_to_pandas(X), transf_df, check_dtype=False)


def test_column_names_are_numbers(df_numeric_columns):
    # integer column names are not supported by polars - pandas only.
    encoder = CountEncoder(
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
    # pandas category dtype is not a polars concept - pandas only.
    encoder = CountEncoder(encoding_method="count", variables=["var_A"])
    X = encoder.fit_transform(df_enc_category_dtypes)

    # expected result
    transf_df = df_enc_category_dtypes.copy()
    transf_df["var_A"] = [6] * 6 + [10] * 10 + [4] * 4
    # transform params
    pd.testing.assert_frame_equal(X, transf_df, check_dtype=False)
    assert X["var_A"].dtypes == int

    encoder = CountEncoder(encoding_method="frequency", variables=["var_A"])
    X = encoder.fit_transform(df_enc_category_dtypes)
    assert X["var_A"].dtypes == float


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_no_unseen(make_df):
    df = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = CountEncoder()
    enc.fit(df)
    dft = enc.transform(df)
    pd.testing.assert_frame_equal(
        _to_pandas(enc.inverse_transform(dft)), _to_pandas(df)
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_ignore_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    df3 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", None]})
    enc = CountEncoder(unseen="ignore")
    enc.fit(df1)
    dft = enc.transform(df2)
    pd.testing.assert_frame_equal(_to_pandas(enc.inverse_transform(dft)), df3)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_encode_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    df3 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", None]})
    enc = CountEncoder(unseen="encode")
    enc.fit(df1)
    dft = enc.transform(df2)
    pd.testing.assert_frame_equal(_to_pandas(enc.inverse_transform(dft)), df3)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_raises_non_fitted_error(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = CountEncoder()

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)

    df1_na = make_df({"words": ["dog", "dog", "cat", "cat", "cat", None]})

    with pytest.raises(ValueError):
        enc.fit(df1_na)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_count_frequency_encoder_is_deprecated(make_df):
    """CountFrequencyEncoder should emit a FutureWarning and still work."""
    X = make_df({"var_A": ["A"] * 6 + ["B"] * 2 + ["C"] * 2})

    with pytest.warns(FutureWarning, match="CountFrequencyEncoder was deprecated"):
        enc = CountFrequencyEncoder(encoding_method="count")
    assert isinstance(enc, CountEncoder)

    enc_new = CountEncoder(encoding_method="count")

    pd.testing.assert_frame_equal(
        _to_pandas(enc.fit_transform(X)), _to_pandas(enc_new.fit_transform(X))
    )
