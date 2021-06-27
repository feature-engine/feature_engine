import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import LogCpTransformer

_test_automatic_find_variables = [
    (
        "e",
        [
            3.6635616461296463,
            3.6888794541139363,
            3.6375861597263857,
            3.6109179126442243,
        ],
        [
            0.9162907318741551,
            0.8754687373539001,
            0.8329091229351039,
            0.7884573603642703,
        ],
    ),
    (
        "10",
        [1.591064607026499, 1.6020599913279625, 1.5797835966168101, 1.568201724066995],
        [
            0.3979400086720376,
            0.3802112417116061,
            0.36172783601759284,
            0.3424226808222063,
        ],
    ),
]


@pytest.mark.parametrize("log_base, exp_age, exp_marks", _test_automatic_find_variables)
def test_logcp_base_plus_automatically_find_variables(
    log_base, exp_age, exp_marks, df_vartypes
):
    transformer = LogCpTransformer(base=log_base, variables=None)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = exp_age
    transf_df["Marks"] = exp_marks

    # test init params
    assert transformer.base == log_base
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


@pytest.mark.parametrize("log_base, exp_age, exp_marks", _test_automatic_find_variables)
def test_log_base_plus_user_passes_var_list(log_base, exp_age, exp_marks, df_vartypes):
    transformer = LogCpTransformer(base=log_base, variables="Age")
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = exp_age

    # test init params
    assert transformer.base == log_base
    assert transformer.variables == "Age"
    # test fit attr
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


def test_error_if_base_value_not_allowed():
    with pytest.raises(ValueError):
        LogCpTransformer(base="other")


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = LogCpTransformer()
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = LogCpTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_if_df_contains_negative_values(df_vartypes):
    # test error when data contains negative values
    df_neg = df_vartypes.copy()
    df_neg.loc[1, "Age"] = -df_vartypes["Age"].min() - 2

    # when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = LogCpTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_neg)


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = LogCpTransformer()
        transformer.transform(df_vartypes)
