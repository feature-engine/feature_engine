import numpy as np
import pandas as pd
import pytest

from feature_engine.encoding import DecisionTreeEncoder


# init parameters
@pytest.mark.parametrize("enc_method", ["count", False, 1])
def test_error_if_encoding_method_not_permitted_value(enc_method):
    msg = (
        "`encoding_method` takes only values 'ordered' and 'arbitrary'."
        f" Got {enc_method} instead."
    )
    with pytest.raises(ValueError) as record:
        DecisionTreeEncoder(encoding_method=enc_method)
    assert str(record.value) == msg


@pytest.mark.parametrize(
    "unseen", ["string", False, ("raise", "ignore"), ["ignore"], np.nan]
)
def test_error_if_unseen_gets_not_permitted_value(unseen):
    msg = (
        'Parameter `unseen` takes only values "ignore", "raise", "encode". '
        f"Got {unseen} instead."
    )
    with pytest.raises(ValueError) as record:
        DecisionTreeEncoder(unseen=unseen)
    str(record.value) == msg


def test_error_if_unseen_is_encode_and_fill_value_is_none():
    msg = (
        "When `unseen='encode'` you need to pass a number to `fill_value`. "
        f"Got {None} instead."
    )
    with pytest.raises(ValueError) as record:
        DecisionTreeEncoder(unseen="encode", fill_value=None)
    str(record.value) == msg


@pytest.mark.parametrize(
    "params",
    [
        ("arbitrary", True, "raise", None),
        ("ordered", False, "ignore", 1),
        ("ordered", False, "encode", 0.1),
    ],
)
def test_init_param_assignment(params):
    DecisionTreeEncoder(
        encoding_method=params[0],
        ignore_format=params[1],
        unseen=params[2],
        fill_value=params[3],
    )


# fit attributes
def test_encoding_dictionary(df_enc):
    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])

    # Tree: var_A <= 1.5 -> 0.25 else 0.5
    # Tree: var_B <= 0.5 -> 0.2 else 0.4
    expected_encodings = {
        "var_A": {"A": 0.25, "B": 0.25, "C": 0.5},
        "var_B": {"A": 0.2, "B": 0.4, "C": 0.4}
    }
    assert encoder.encoder_dict_ == expected_encodings


def test_classification(df_enc):
    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    transf_df = df_enc.copy()
    transf_df["var_A"] = [0.25] * 16 + [0.5] * 4  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.2] * 10 + [0.4] * 10  # Tree: var_B <= 0.5 -> 0.2 else 0.4
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_regression(df_enc):
    random = np.random.RandomState(42)
    y = random.normal(0, 0.1, len(df_enc))
    encoder = DecisionTreeEncoder(
        regression=True,
        random_state=random,
    )
    encoder.fit(df_enc[["var_A", "var_B"]], y)
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    transf_df = df_enc.copy()
    transf_df["var_A"] = (
        [0.034348] * 6 + [-0.024679] * 10 + [-0.075473] * 4
    )  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.044806] * 10 + [-0.079066] * 10
    pd.testing.assert_frame_equal(X.round(6), transf_df[["var_A", "var_B"]])


def test_fit_raises_error_if_df_contains_na(df_enc_na):
    # test case 4: when dataset contains na, fit method
    encoder = DecisionTreeEncoder(regression=False)
    with pytest.raises(ValueError) as record:
        encoder.fit(df_enc_na[["var_A", "var_B"]], df_enc_na["target"])
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    assert str(record.value) == msg


def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na):
    # test case 4: when dataset contains na, transform method
    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    with pytest.raises(ValueError) as record:
        encoder.transform(df_enc_na[["var_A", "var_B"]])
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    assert str(record.value) == msg


def test_classification_ignore_format(df_enc_numeric):
    encoder = DecisionTreeEncoder(
        regression=False,
        ignore_format=True,
    )
    encoder.fit(df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"])
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

    transf_df = df_enc_numeric.copy()
    transf_df["var_A"] = [0.25] * 16 + [0.5] * 4  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.2] * 10 + [0.4] * 10  # Tree: var_B <= 0.5 -> 0.2 else 0.4
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_regression_ignore_format(df_enc_numeric):
    random = np.random.RandomState(42)
    y = random.normal(0, 0.1, len(df_enc_numeric))
    encoder = DecisionTreeEncoder(
        regression=True,
        random_state=random,
        ignore_format=True,
    )
    encoder.fit(df_enc_numeric[["var_A", "var_B"]], y)
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

    transf_df = df_enc_numeric.copy()
    transf_df["var_A"] = (
        [0.034348] * 6 + [-0.024679] * 10 + [-0.075473] * 4
    )  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.044806] * 10 + [-0.079066] * 10
    pd.testing.assert_frame_equal(X.round(6), transf_df[["var_A", "var_B"]])


def test_variables_cast_as_category(df_enc_category_dtypes):
    df = df_enc_category_dtypes.copy()
    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    transf_df = df.copy()
    transf_df["var_A"] = [0.25] * 16 + [0.5] * 4  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.2] * 10 + [0.4] * 10  # Tree: var_B <= 0.5 -> 0.2 else 0.4
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes == float


def test_error_when_regression_is_true_and_target_is_binary(df_enc):
    encoder = DecisionTreeEncoder(regression=True)
    with pytest.raises(ValueError) as record:
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    msg = (
        "Trying to fit a regression to a binary target is not "
        "allowed by this transformer. Check the target values "
        "or set regression to False."
    )
    assert str(record.value) == msg


def test_error_when_regression_is_false_and_target_is_continuous(df_enc):
    random = np.random.RandomState(42)
    y = random.normal(0, 10, len(df_enc))
    encoder = DecisionTreeEncoder(regression=False)
    # the error message comes from sklearn api - won't test
    with pytest.raises(ValueError):
        encoder.fit(df_enc[["var_A", "var_B"]], y)


def test_inverse_transform_raises_not_implemented_error(df_enc):
    random = np.random.RandomState(42)
    y = random.normal(0, 10, len(df_enc))
    encoder = DecisionTreeEncoder(regression=True).fit(df_enc[["var_A", "var_B"]], y)
    with pytest.raises(NotImplementedError) as record:
        encoder.inverse_transform(df_enc[["var_A", "var_B"]])
    msg = "inverse_transform is not implemented for this transformer."
    assert str(record.value) == msg


@pytest.mark.parametrize(
    "grid",
    [None, {"max_depth": [1, 2, 3]}, {"max_depth": [1, 2], "estimators": [10, 12]}],
)
def test_assigns_param_grid(grid):
    encoder = DecisionTreeEncoder(param_grid=grid)
    if grid is None:
        assert encoder._assign_param_grid() == {"max_depth": [1, 2, 3, 4]}
    else:
        assert encoder._assign_param_grid() == grid


def test_unseen_is_encode(df_enc):
    encoder = DecisionTreeEncoder(unseen="encode", regression=False, fill_value=-1)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])

    X_unseen_input = pd.DataFrame(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )

    X_unseen_output = pd.DataFrame(
        {
            "var_A": [0.25, -1, -1],
            "var_B": [0.4, -1, -1],
        }
    )

    Xt = encoder.transform(X_unseen_input)
    pd.testing.assert_frame_equal(Xt, X_unseen_output)


def test_unseen_is_ignore(df_enc):
    encoder = DecisionTreeEncoder(unseen="ignore", regression=False)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])

    X_unseen_input = pd.DataFrame(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )

    X_unseen_output = pd.DataFrame(
        {
            "var_A": [0.25, np.nan, np.nan],
            "var_B": [0.4, np.nan, np.nan],
        }
    )

    Xt = encoder.transform(X_unseen_input)
    pd.testing.assert_frame_equal(Xt, X_unseen_output)


def test_fit_errors_if_new_cat_values_and_unseen_is_raise_param(df_enc):
    encoder = DecisionTreeEncoder(unseen="raise", regression=False)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = pd.DataFrame(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )
    # new categories will raise an error
    with pytest.raises(ValueError) as record:
        encoder.transform(X)
    var_ls = "var_A, var_B"
    msg = (
        "During the encoding, NaN values were introduced in the feature(s) "
        f"{var_ls}."
    )
    assert str(record.value) == msg
