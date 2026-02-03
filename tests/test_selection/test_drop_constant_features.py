import numpy as np
import pandas as pd
import pytest

from feature_engine.selection import DropConstantFeatures


@pytest.fixture(scope="module")
def df_constant_features():
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
        "const_feat_num": [1, 1, 1, 1],
        "const_feat_cat": ["a", "a", "a", "a"],
        "quasi_feat_num": [1, 1, 1, 2],
        "quasi_feat_cat": ["a", "a", "a", "b"],
    }

    df = pd.DataFrame(data)

    return df


def test_drop_constant_features(df_constant_features):
    transformer = DropConstantFeatures(tol=1, variables=None)
    X = transformer.fit_transform(df_constant_features)

    # expected result
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
            "quasi_feat_num": [1, 1, 1, 2],
            "quasi_feat_cat": ["a", "a", "a", "b"],
        }
    )

    # fit attribute
    assert transformer.features_to_drop_ == ["const_feat_num", "const_feat_cat"]

    # transform output
    pd.testing.assert_frame_equal(X, df)


def test_drop_constant_and_quasiconstant_features(df_constant_features):
    transformer = DropConstantFeatures(tol=0.7, variables=None)
    X = transformer.fit_transform(df_constant_features)

    # expected result
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
        }
    )

    # fit attr
    assert transformer.features_to_drop_ == [
        "const_feat_num",
        "const_feat_cat",
        "quasi_feat_num",
        "quasi_feat_cat",
    ]

    # transform params
    pd.testing.assert_frame_equal(X, df)


def test_drop_constant_features_with_list_of_variables(df_constant_features):
    # test case 3: drop features showing threshold more than 0.7 with variable list
    transformer = DropConstantFeatures(
        tol=0.7, variables=["Name", "const_feat_num", "quasi_feat_num"]
    )
    X = transformer.fit_transform(df_constant_features)

    # expected result
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
            "const_feat_cat": ["a", "a", "a", "a"],
            "quasi_feat_cat": ["a", "a", "a", "b"],
        }
    )

    # fit attr
    assert transformer.features_to_drop_ == ["const_feat_num", "quasi_feat_num"]

    # transform params
    pd.testing.assert_frame_equal(X, df)


@pytest.mark.parametrize("tol", [2, "hola", False])
def test_error_if_tol_value_not_allowed(tol):
    # test case 5: threshold not between 0 and 1
    with pytest.raises(ValueError):
        DropConstantFeatures(tol=tol)


@pytest.mark.parametrize("tol", [1, 0, 0.5, 0.7])
def test_tol_init_param(tol):
    sel = DropConstantFeatures(tol=tol)
    assert sel.tol == tol


@pytest.mark.parametrize("missing", [2, "hola", False])
def test_error_if_missing_values_not_permitted(missing):
    # test case 5: threshold not between 0 and 1
    with pytest.raises(ValueError):
        DropConstantFeatures(missing_values=missing)


def test_error_if_all_constant_and_quasi_constant_features():
    # test case 6: when input contains all constant features
    with pytest.raises(ValueError):
        DropConstantFeatures().fit(pd.DataFrame({"col1": [1, 1, 1], "col2": [1, 1, 1]}))

    # test case 7: when input contains all constant and quasi constant features
    with pytest.raises(ValueError):
        transformer = DropConstantFeatures(tol=0.7)
        transformer.fit_transform(
            pd.DataFrame(
                {
                    "col1": [1, 1, 1, 1],
                    "col2": [1, 1, 1, 1],
                    "col3": [1, 1, 1, 2],
                    "col4": [1, 1, 1, 2],
                }
            )
        )


def test_missing_values_param_functionality():

    df = {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
        "const_feat_num": [1, 1, 1, np.nan],
        "const_feat_cat": ["a", "a", "a", "a"],
        "quasi_feat_num": [1, 1, 1, 2],
        "quasi_feat_cat": ["a", "a", "a", np.nan],
    }
    df = pd.DataFrame(df)

    # test raises error if there is na
    transformer = DropConstantFeatures(missing_values="raise")
    with pytest.raises(ValueError):
        transformer.fit(df)

    # test ignores na
    transformer = DropConstantFeatures(missing_values="ignore").fit(df)
    constant = ["const_feat_num", "const_feat_cat", "quasi_feat_cat"]
    assert transformer.features_to_drop_ == constant
    pd.testing.assert_frame_equal(df.drop(constant, axis=1), transformer.transform(df))

    # test includes na
    transformer = DropConstantFeatures(tol=0.7, missing_values="include").fit(df)
    qconstant = ["const_feat_num", "const_feat_cat", "quasi_feat_num", "quasi_feat_cat"]
    assert transformer.features_to_drop_ == qconstant
    pd.testing.assert_frame_equal(df.drop(qconstant, axis=1), transformer.transform(df))
