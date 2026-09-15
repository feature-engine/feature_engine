import re
import warnings

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import CountEncoder, CountFrequencyEncoder
from tests.backend_helpers import null_count, frame_to_dict

DATA_VARTYPES = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
    "dob": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
}


# init parameters
@pytest.mark.parametrize("enc_method", ["arbitrary", False, 1])
def test_error_if_encoding_method_not_permitted_value(enc_method):
    msg = (
        "encoding_method takes only values 'count' and 'frequency'. "
        f"Got {enc_method} instead."
    )
    with pytest.raises(ValueError, match=msg):
        CountEncoder(encoding_method=enc_method)


@pytest.mark.parametrize(
    "errors", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_gets_not_permitted_value(errors):
    msg = (
        "Parameter `unseen` takes only values ignore, raise, encode. "
        f"Got {errors} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
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
def test_encode_1_variable_with_counts(make_df, data_enc):
    # test case 1: 1 variable, counts
    encoder = CountEncoder(encoding_method="count", variables=["var_A"])
    X = encoder.fit_transform(make_df(data_enc))

    # fit params
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {"A": 6, "B": 10, "C": 4}}
    assert encoder.n_features_in_ == 3
    # transform params
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {
        "var_A": [6] * 6 + [10] * 10 + [4] * 4,
        "var_B": data_enc["var_B"],
        "target": data_enc["target"],
    }


def test_automatically_select_variables_encode_with_frequency(make_df, data_enc):
    # test case 2: automatically select variables, frequency
    encoder = CountEncoder(encoding_method="frequency", variables=None)
    X = encoder.fit_transform(make_df(data_enc))

    # fit params
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.3, "B": 0.5, "C": 0.2},
        "var_B": {"A": 0.5, "B": 0.3, "C": 0.2},
    }
    assert encoder.n_features_in_ == 3
    # transform params
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {
        "var_A": [0.3] * 6 + [0.5] * 10 + [0.2] * 4,
        "var_B": [0.5] * 10 + [0.3] * 6 + [0.2] * 4,
        "target": data_enc["target"],
    }


def test_encoding_when_nan_in_fit_df(make_df, data_enc):
    encoder = CountEncoder(
        encoding_method="frequency",
        missing_values="ignore",
    )
    encoder.fit(make_df(data_enc))

    X = encoder.transform(
        make_df({"var_A": ["A", None], "var_B": ["A", None], "target": [1, 0]})
    )

    # transform params
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {
        "var_A": [0.3, None],
        "var_B": [0.5, None],
        "target": [1, 0],
    }


def test_warning_when_df_contains_unseen_categories(
    make_df, data_enc, data_enc_rare
):
    # dataset to be transformed contains categories not present in
    # training dataset (unseen categories), unseen set to ignore.
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when unseen equals 'ignore'
    encoder = CountEncoder(unseen="ignore")
    encoder.fit(make_df(data_enc))
    with pytest.warns(UserWarning, match=re.escape(msg)):
        encoder.transform(make_df(data_enc_rare))


def test_error_when_df_contains_unseen_categories(make_df, data_enc, data_enc_rare):
    # dataset to be transformed contains categories not present in
    # training dataset (unseen categories), unseen set to raise.
    df_enc = make_df(data_enc)
    df_enc_rare = make_df(data_enc_rare)

    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    encoder = CountEncoder(unseen="raise")
    encoder.fit(df_enc)

    # check for exception when unseen equals 'raise'
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(df_enc_rare)


@pytest.mark.parametrize("errors", ["raise", "ignore", "encode"])
def test_fit_raises_error_if_df_contains_na(errors, make_df, data_enc_na):
    # test case 4: when dataset contains na, fit method
    encoder = CountEncoder(unseen=errors)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.fit(make_df(data_enc_na))


@pytest.mark.parametrize("errors", ["raise", "ignore", "encode"])
def test_transform_raises_error_if_df_contains_na(
    errors, make_df, data_enc, data_enc_na
):
    # test case 4: when dataset contains na, transform method
    encoder = CountEncoder(unseen=errors)
    encoder.fit(make_df(data_enc))
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(make_df(data_enc_na))


def test_zero_encoding_for_unseen_categories_if_unseen_is_encode(make_df):
    df_fit = make_df(
        {"col1": ["a", "a", "b", "a", "c"], "col2": ["1", "2", "3", "1", "2"]}
    )
    df_transform = make_df(
        {"col1": ["a", "d", "b", "a", "c"], "col2": ["1", "2", "3", "1", "4"]}
    )

    # count encoding
    encoder = CountEncoder(unseen="encode").fit(df_fit)
    # unseen categories are encoded without raising or warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = encoder.transform(df_transform)
    assert isinstance(result, make_df)

    # check that no NaNs are added
    assert null_count(result, "col1") == 0
    assert null_count(result, "col2") == 0

    # check that the counts are correct
    assert frame_to_dict(result) == {"col1": [3, 0, 1, 3, 1], "col2": [2, 2, 1, 2, 0]}

    # with frequency
    encoder = CountEncoder(encoding_method="frequency", unseen="encode").fit(df_fit)
    # unseen categories are encoded without raising or warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = encoder.transform(df_transform)
    assert isinstance(result, make_df)

    # check that no NaNs are added
    assert null_count(result, "col1") == 0
    assert null_count(result, "col2") == 0

    # check that the frequencies are correct
    assert frame_to_dict(result) == {
        "col1": [0.6, 0, 0.2, 0.6, 0.2],
        "col2": [0.4, 0.4, 0.2, 0.4, 0],
    }


def test_nan_encoding_for_new_categories_if_unseen_is_ignore(make_df):
    df_fit = make_df(
        {"col1": ["a", "a", "b", "a", "c"], "col2": ["1", "2", "3", "1", "2"]}
    )
    df_transf = make_df(
        {"col1": ["a", "d", "b", "a", "c"], "col2": ["1", "2", "3", "1", "4"]}
    )
    encoder = CountEncoder(unseen="ignore").fit(df_fit)
    result = encoder.transform(df_transf)
    assert isinstance(result, make_df)

    # check that 1 NaN is added per variable
    assert null_count(result, "col1") == 1
    assert null_count(result, "col2") == 1

    # check that the counts are correct for both new and old
    assert frame_to_dict(result) == {
        "col1": [3, None, 1, 3, 1],
        "col2": [2, 2, 1, 2, None],
    }


def test_ignore_variable_format_with_frequency(make_df):
    encoder = CountEncoder(
        encoding_method="frequency", variables=None, ignore_format=True
    )
    X = encoder.fit_transform(make_df(DATA_VARTYPES))

    # fit params
    assert encoder.variables_ == ["Name", "City", "Age", "Marks", "dob"]
    assert encoder.n_features_in_ == 5
    # transform params
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {
        "Name": [0.25, 0.25, 0.25, 0.25],
        "City": [0.25, 0.25, 0.25, 0.25],
        "Age": [0.25, 0.25, 0.25, 0.25],
        "Marks": [0.25, 0.25, 0.25, 0.25],
        "dob": [0.25, 0.25, 0.25, 0.25],
    }


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


def test_inverse_transform_when_no_unseen(make_df):
    words = ["dog", "dog", "cat", "cat", "cat", "bird"]
    df = make_df({"words": words})
    enc = CountEncoder()
    enc.fit(df)
    dft = enc.transform(df)
    X = enc.inverse_transform(dft)
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"words": words}


def test_inverse_transform_when_ignore_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    enc = CountEncoder(unseen="ignore")
    enc.fit(df1)
    dft = enc.transform(df2)
    X = enc.inverse_transform(dft)
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"words": ["dog", "dog", "cat", "cat", "cat", None]}


def test_inverse_transform_when_encode_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    enc = CountEncoder(unseen="encode")
    enc.fit(df1)
    dft = enc.transform(df2)
    X = enc.inverse_transform(dft)
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"words": ["dog", "dog", "cat", "cat", "cat", None]}


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


def test_count_frequency_encoder_is_deprecated(make_df):
    """CountFrequencyEncoder should emit a FutureWarning and still work."""
    X = make_df({"var_A": ["A"] * 6 + ["B"] * 2 + ["C"] * 2})

    with pytest.warns(FutureWarning, match="CountFrequencyEncoder was deprecated"):
        enc = CountFrequencyEncoder(encoding_method="count")
    assert isinstance(enc, CountEncoder)

    enc_new = CountEncoder(encoding_method="count")

    X_old = enc.fit_transform(X)
    X_new = enc_new.fit_transform(X)
    assert isinstance(X_old, make_df)
    assert isinstance(X_new, make_df)
    assert (
        frame_to_dict(X_old)
        == frame_to_dict(X_new)
        == {"var_A": [6] * 6 + [2] * 2 + [2] * 2}
    )
