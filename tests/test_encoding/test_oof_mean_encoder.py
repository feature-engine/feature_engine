import numpy as np
import pandas as pd
import pytest
from numpy import nan
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from feature_engine.encoding import OOFMeanEncoder


# test init params
@pytest.mark.parametrize(
    "params", [("raise", True, "auto", 3, True, 1, False), ("ignore", False, 1, 5, False, None, True)]
)
def test_init_param_assignment(params):
    enc = OOFMeanEncoder(
        missing_values=params[0],
        ignore_format=params[1],
        smoothing=params[2],
        cv=params[3],
        shuffle=params[4],
        random_state=params[5],
        stratify=params[6],
    )
    assert enc.missing_values == params[0]
    assert enc.ignore_format == params[1]
    assert enc.smoothing == params[2]
    assert enc.cv == params[3]
    assert enc.shuffle == params[4]
    assert enc.random_state == params[5]
    assert enc.stratify == params[6]


@pytest.mark.parametrize(
    "errors", ["invalid", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_gets_not_permitted_value(errors):
    with pytest.raises(ValueError):
        OOFMeanEncoder(unseen=errors)


@pytest.mark.parametrize("smoothing", ["hello", ["auto"], -1, -0.5])
def test_raises_error_when_not_allowed_smoothing_param_in_init(smoothing):
    with pytest.raises(ValueError):
        OOFMeanEncoder(smoothing=smoothing)


@pytest.mark.parametrize("cv", [0, 1, -2])
def test_raises_error_when_cv_less_than_2(cv):
    with pytest.raises(ValueError):
        OOFMeanEncoder(cv=cv)


@pytest.mark.parametrize("shuffle", ["true", 1, None, []])
def test_raises_error_when_shuffle_not_bool(shuffle):
    with pytest.raises(ValueError):
        OOFMeanEncoder(shuffle=shuffle)


@pytest.mark.parametrize("stratify", ["true", 1, None, []])
def test_raises_error_when_stratify_not_bool(stratify):
    with pytest.raises(ValueError):
        OOFMeanEncoder(stratify=stratify)


# fit and transform
def test_fit_and_transform_single_variable(df_enc):
    encoder = OOFMeanEncoder(variables=["var_A"])
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc.copy()
    transf_df["var_A"] = [
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.5,
        0.5,
        0.5,
        0.5,
    ]

    assert encoder.variables == ["var_A"]
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.3333333333333333, "B": 0.2, "C": 0.5}
    }
    assert encoder.n_features_in_ == 2
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_automatically_find_variables(df_enc):
    encoder = OOFMeanEncoder(variables=None)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    assert encoder.variables_ == ["var_A", "var_B"]
    assert "var_A" in encoder.encoder_dict_
    assert "var_B" in encoder.encoder_dict_


def test_oof_encoding_prevents_leakage():
    # Construct a dataset where category 'A' has target 1 in fold 1 and 0 in fold 2.
    # Out of fold encoding should assign 0 to fold 1 and 1 to fold 2.
    df = pd.DataFrame(
        {
            "cat": ["A", "A", "A", "A"],
            "num": [10, 20, 30, 40],
        }
    )
    y = pd.Series([1, 1, 0, 0])

    encoder = OOFMeanEncoder(variables=["cat"], cv=2, shuffle=False)
    X_oof = encoder.fit_transform(df, y)

    # Fold 1 (rows 0, 1): trained on fold 2 (target [0, 0]), so mapped to 0.0
    # Fold 2 (rows 2, 3): trained on fold 1 (target [1, 1]), so mapped to 1.0
    expected_oof = [0.0, 0.0, 1.0, 1.0]
    assert np.allclose(X_oof["cat"].tolist(), expected_oof)

    # Subsequent transform on new data uses full dataset mapping (mean = 0.5)
    X_test = encoder.transform(df)
    assert np.allclose(X_test["cat"].tolist(), [0.5, 0.5, 0.5, 0.5])


def test_cv_as_custom_splitter(df_enc):
    cv = KFold(n_splits=4, shuffle=False)
    encoder = OOFMeanEncoder(cv=cv, unseen="encode")
    X_tr = encoder.fit_transform(df_enc[["var_A", "var_B"]], df_enc["target"])

    assert encoder.n_features_in_ == 2
    assert len(X_tr) == len(df_enc)
    assert not X_tr.isnull().any().any()


def test_cv_stratified(df_enc):
    cv = StratifiedKFold(n_splits=2, shuffle=False)
    encoder = OOFMeanEncoder(cv=cv, stratify=True, unseen="encode")
    X_tr = encoder.fit_transform(df_enc[["var_A", "var_B"]], df_enc["target"])

    assert len(X_tr) == len(df_enc)
    assert not X_tr.isnull().any().any()


def test_stratify_raises_error_on_continuous_target():
    df = pd.DataFrame({"cat": ["A", "B", "A", "B", "A", "B"]})
    y = pd.Series([1.23, 4.56, 7.89, 0.12, 3.45, 6.78])

    encoder = OOFMeanEncoder(cv=2, stratify=True)
    with pytest.raises(ValueError):
        encoder.fit_transform(df, y)


def test_smoothing_fixed(df_enc):
    encoder = OOFMeanEncoder(
        variables=["var_A"], smoothing=10, cv=2, shuffle=False, unseen="encode"
    )
    X_oof = encoder.fit_transform(df_enc[["var_A", "var_B"]], df_enc["target"])
    assert len(X_oof) == len(df_enc)
    assert not X_oof["var_A"].isnull().any()


def test_smoothing_auto(df_enc):
    encoder = OOFMeanEncoder(
        variables=["var_A"], smoothing="auto", cv=2, shuffle=False, unseen="encode"
    )
    X_oof = encoder.fit_transform(df_enc[["var_A", "var_B"]], df_enc["target"])
    assert len(X_oof) == len(df_enc)
    assert not X_oof["var_A"].isnull().any()


def test_unseen_encode():
    # If a category is only in validation fold, unseen='encode' should fill with prior
    df = pd.DataFrame(
        {
            "cat": ["A", "A", "B", "C"],
        }
    )
    y = pd.Series([1, 0, 1, 0])

    encoder = OOFMeanEncoder(variables=["cat"], cv=2, shuffle=False, unseen="encode")
    X_oof = encoder.fit_transform(df, y)
    assert not X_oof["cat"].isnull().any()

    # Inference with unseen category
    df_test = pd.DataFrame({"cat": ["D"]})
    X_test = encoder.transform(df_test)
    assert np.allclose(X_test["cat"].iloc[0], encoder.y_prior_)


def test_unseen_raise():
    df_train = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
    y_train = pd.Series([1, 0, 1, 0])
    encoder = OOFMeanEncoder(unseen="raise")
    encoder.fit(df_train, y_train)

    df_test = pd.DataFrame({"cat": ["C"]})
    with pytest.raises(ValueError) as record:
        encoder.transform(df_test)
    assert "During the encoding, NaN values were introduced" in str(record.value)


def test_missing_values_raise_and_ignore():
    df = pd.DataFrame({"cat": ["A", "B", nan, "B"], "target": [1, 0, 1, 0]})

    encoder_raise = OOFMeanEncoder(missing_values="raise")
    with pytest.raises(ValueError):
        encoder_raise.fit(df[["cat"]], df["target"])

    with pytest.raises(ValueError):
        encoder_raise.fit_transform(df[["cat"]], df["target"])

    encoder_ignore = OOFMeanEncoder(missing_values="ignore", cv=2, shuffle=False)
    encoder_ignore.fit(df[["cat"]], df["target"])
    X_tr = encoder_ignore.transform(df[["cat"]])
    assert np.isnan(X_tr.loc[2, "cat"])


def test_ignore_format_numerical_variables(df_enc_numeric):
    encoder = OOFMeanEncoder(
        ignore_format=True, cv=2, shuffle=False, unseen="encode"
    )
    X_tr = encoder.fit_transform(
        df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"]
    )
    assert encoder.variables_ == ["var_A", "var_B"]
    assert not X_tr.isnull().any().any()


def test_unseen_ignore_warns_when_fold_has_unseen_category():
    # Category 'C' only exists in fold 2. With unseen='ignore', fold 2 will have NaNs for 'A',
    # and fold 1 will have NaNs for 'C', which triggers a warning.
    df = pd.DataFrame(
        {
            "cat": ["A", "A", "B", "B", "C", "C"],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])

    encoder = OOFMeanEncoder(cv=2, shuffle=False, unseen="ignore")
    with pytest.warns(UserWarning, match="During the encoding, NaN values were introduced"):
        X_oof = encoder.fit_transform(df, y)
    assert X_oof["cat"].isnull().any()


def test_return_empty():
    df = pd.DataFrame({"num1": [1, 2, 3], "num2": [4, 5, 6]})
    y = pd.Series([1, 0, 1])
    encoder = OOFMeanEncoder(return_empty=True)
    encoder.fit(df, y)
    assert encoder.variables_ == []
    pd.testing.assert_frame_equal(encoder.transform(df), df)
    pd.testing.assert_frame_equal(encoder.fit_transform(df, y), df)


def test_preserves_non_standard_dataframe_index():
    df = pd.DataFrame(
        {"cat": ["A", "B", "A", "B", "A", "B"]},
        index=[105, 42, 999, 12, 50, 77],
    )
    y = pd.Series([1, 0, 1, 0, 1, 0], index=[105, 42, 999, 12, 50, 77])

    encoder = OOFMeanEncoder(cv=3, shuffle=False)
    X_oof = encoder.fit_transform(df, y)

    pd.testing.assert_index_equal(X_oof.index, df.index)
    X_test = encoder.transform(df)
    pd.testing.assert_index_equal(X_test.index, df.index)


def test_inverse_transform():
    df = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    enc = OOFMeanEncoder()
    enc.fit(df, y)
    dft = enc.transform(df)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df)

    # raises NotImplementedError when unseen='encode'
    enc_encode = OOFMeanEncoder(unseen="encode")
    enc_encode.fit(df, y)
    dft_encode = enc_encode.transform(df)
    with pytest.raises(NotImplementedError):
        enc_encode.inverse_transform(dft_encode)


def test_inverse_transform_raises_non_fitted_error():
    df = pd.DataFrame({"words": ["dog", "cat"]})
    enc = OOFMeanEncoder()
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df)


def test_pipeline_compatibility(df_enc):
    pipe = Pipeline(
        [
            ("oof", OOFMeanEncoder(variables=["var_A"], cv=2, shuffle=False)),
        ]
    )
    X_tr = pipe.fit_transform(df_enc[["var_A", "var_B"]], df_enc["target"])
    assert len(X_tr) == len(df_enc)

    X_test = pipe.transform(df_enc[["var_A", "var_B"]])
    assert len(X_test) == len(df_enc)
