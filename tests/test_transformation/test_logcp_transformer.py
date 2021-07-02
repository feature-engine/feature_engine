import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import LogCpTransformer

_age_inverse_transform = [20.0, 21.0, 19.0, 18.0]

_learned_C = {"Age": 19.0, "Marks": 1.6}  # [19, 1.6]

_params_test_automatic_find_variables = [
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
        _learned_C,
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
        _learned_C,
    ),
]

_params_test_inverse_transform = [
    ("e", _age_inverse_transform, _learned_C["Age"]),
    ("10", _age_inverse_transform, _learned_C["Age"]),
]


@pytest.mark.parametrize(
    "log_base, exp_age, exp_marks, learned_c", _params_test_automatic_find_variables
)
def test_logcp_base_plus_automatically_find_variables(
    log_base, exp_age, exp_marks, learned_c, df_vartypes
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
    assert transformer.C == "auto"
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    assert transformer.C_ == learned_c
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


@pytest.mark.parametrize(
    "log_base, exp_age, exp_marks, learned_c", _params_test_automatic_find_variables
)
def test_log_base_plus_user_passes_var_list(
    log_base, exp_age, exp_marks, learned_c, df_vartypes
):
    user_var = "Age"
    transformer = LogCpTransformer(base=log_base, variables=user_var)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df[user_var] = exp_age

    # test init params
    assert transformer.base == log_base
    assert transformer.variables == user_var
    assert transformer.C == "auto"
    # test fit attr
    assert transformer.variables_ == [user_var]
    assert transformer.n_features_in_ == 5
    assert transformer.C_ == {user_var: learned_c["Age"]}
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


@pytest.mark.parametrize("log_base, exp_age, c_age", _params_test_inverse_transform)
def test_inverse_transform(log_base, exp_age, c_age, df_vartypes):

    user_var = "Age"
    transformer = LogCpTransformer(base=log_base, variables=user_var)
    X_t = transformer.fit_transform(df_vartypes)
    X_dt = transformer.inverse_transform(X_t)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df[user_var] = exp_age

    # test init params
    assert transformer.base == log_base
    assert transformer.C == "auto"
    # test fit attr
    assert transformer.variables_ == [user_var]
    assert transformer.n_features_in_ == 5
    assert transformer.C_ == {user_var: c_age}
    # test transform output
    pd.testing.assert_frame_equal(X_dt, transf_df)


def test_error_if_base_value_not_allowed():
    with pytest.raises(ValueError):
        LogCpTransformer(base="other")


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = LogCpTransformer()
        transformer.fit(df_na)


def test_fit_raises_error_if_negative_values(df_vartypes):
    # test error: when variable + C contain negative values
    user_var = "Age"
    df_neg = df_vartypes.copy()
    df_neg.loc[2, user_var] = -7

    with pytest.raises(ValueError) as errmsg:
        transformer = LogCpTransformer(base="e", variables=user_var, C=1)
        transformer.fit(df_neg)

    exceptionmsg = errmsg.value.args[0]

    assert (
        exceptionmsg
        == "Some variables contain zero or negative values after addingconstant C, "
        + "can't apply log"
    )


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = LogCpTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_if_df_contains_negative_values(df_vartypes):
    # test error when data contains negative values
    user_var = "Age"
    df_neg = df_vartypes.copy()
    df_neg.loc[1, user_var] = -df_vartypes[user_var].min() - 2

    # when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = LogCpTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_neg)


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = LogCpTransformer()
        transformer.transform(df_vartypes)
