import pandas as pd
import pytest

from feature_engine.encoding import CountFrequencyEncoder


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


def test_error_if_encoding_method_not_permitted_value():
    with pytest.raises(ValueError):
        CountFrequencyEncoder(encoding_method="arbitrary")


def test_error_if_input_df_contains_categories_not_present_in_fit_df(
    df_enc, df_enc_rare
):
    # test case 3: when dataset to be transformed contains categories not present in
    # training dataset

    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when rare_labels equals 'ignore'
    with pytest.warns(UserWarning) as record:
        encoder = CountFrequencyEncoder(errors="ignore")
        encoder.fit(df_enc)
        encoder.transform(df_enc_rare)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record:
        encoder = CountFrequencyEncoder(errors="raise")

        encoder.fit(df_enc)
        encoder.transform(df_enc_rare)

    # check that the error message matches
    assert str(record.value) == msg


def test_fit_raises_error_if_df_contains_na(df_enc_na):
    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = CountFrequencyEncoder()
        encoder.fit(df_enc_na)


def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = CountFrequencyEncoder()
        encoder.fit(df_enc)
        encoder.transform(df_enc_na)


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
        4: pd.date_range("2020-02-24", periods=4, freq="T"),
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


def test_error_if_rare_labels_not_permitted_value():
    with pytest.raises(ValueError):
        CountFrequencyEncoder(errors="empanada")
