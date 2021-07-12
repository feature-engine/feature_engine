import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import LogCpTransformer

_learned_C = {"Age": 19.0, "Marks": 1.6}

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

_params_test_c_is_numerical = [
    (
        "e",
        [3.044522437723423, 3.091042453358316, 2.995732273553991, 2.9444389791664403],
        [
            0.6418538861723947,
            0.5877866649021191,
            0.5306282510621704,
            0.47000362924573563,
        ],
    ),
    (
        "10",
        [
            1.3222192947339193,
            1.3424226808222062,
            1.3010299956639813,
            1.2787536009528289,
        ],
        [
            0.2787536009528289,
            0.25527250510330607,
            0.2304489213782739,
            0.20411998265592482,
        ],
    ),
]


@pytest.mark.parametrize(
    "log_base, exp_age, exp_marks, learned_c", _params_test_automatic_find_variables
)
def test_params_C_is_auto_variables_is_none_all_possible_bases(
    log_base, exp_age, exp_marks, learned_c, df_vartypes
):
    transformer = LogCpTransformer(base=log_base, variables=None, C="auto")
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

    # test inverse_transform output
    Xit = transformer.inverse_transform(X)
    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    pd.testing.assert_frame_equal(Xit, df_vartypes)


@pytest.mark.parametrize(
    "log_base, exp_age, exp_marks, learned_c", _params_test_automatic_find_variables
)
def test_param_C_is_dict_user(log_base, exp_age, exp_marks, learned_c, df_vartypes):
    transformer = LogCpTransformer(base=log_base, C={"Age": 19.0})
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = exp_age

    # test init params
    assert transformer.base == log_base
    assert transformer.variables is None
    assert transformer.C == {"Age": 19.0}
    # test fit attr
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 5
    assert transformer.C_ == {"Age": learned_c["Age"]}
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)

    # test inverse_transform output
    Xit = transformer.inverse_transform(X)
    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    pd.testing.assert_frame_equal(Xit, df_vartypes)


@pytest.mark.parametrize("log_base, exp_age, exp_marks", _params_test_c_is_numerical)
def test_param_C_is_int_variables_is_none(log_base, exp_age, exp_marks, df_vartypes):
    int_constant = 1
    transformer = LogCpTransformer(base=log_base, C=int_constant)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = exp_age
    transf_df["Marks"] = exp_marks

    # test init params
    assert transformer.base == log_base
    assert transformer.variables is None
    assert transformer.C == int_constant
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    assert transformer.C_ == int_constant
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)

    # test inverse_transform output
    Xit = transformer.inverse_transform(X)
    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    pd.testing.assert_frame_equal(Xit, df_vartypes)


@pytest.mark.parametrize("log_base, exp_age, exp_marks", _params_test_c_is_numerical)
def test_param_C_is_float_variables_is_none(log_base, exp_age, exp_marks, df_vartypes):
    float_constant = 1.0
    transformer = LogCpTransformer(base=log_base, C=float_constant)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = exp_age
    transf_df["Marks"] = exp_marks

    # test init params
    assert transformer.base == log_base
    assert transformer.variables is None
    assert transformer.C == float_constant
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    assert transformer.C_ == float_constant
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)

    # test inverse_transform output
    Xit = transformer.inverse_transform(X)
    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    pd.testing.assert_frame_equal(Xit, df_vartypes)


@pytest.mark.parametrize(
    "log_base, exp_age, exp_marks, learned_c", _params_test_automatic_find_variables
)
def test_user_pass_variables(log_base, exp_age, exp_marks, learned_c, df_vartypes):
    user_vars = ["Age", "Marks"]
    transformer = LogCpTransformer(base=log_base, variables=user_vars)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = exp_age
    transf_df["Marks"] = exp_marks

    # test init params
    assert transformer.base == log_base
    assert transformer.variables == user_vars
    assert transformer.C == "auto"
    # test fit attr
    assert transformer.variables_ == user_vars
    assert transformer.n_features_in_ == 5
    assert transformer.C_ == learned_c
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)

    # test inverse_transform output
    Xit = transformer.inverse_transform(X)
    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    pd.testing.assert_frame_equal(Xit, df_vartypes)


def test_error_if_base_value_not_allowed():
    with pytest.raises(ValueError):
        LogCpTransformer(base="other")


def test_raises_error_if_na_in_df(df_na, df_vartypes):
    # when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = LogCpTransformer()
        transformer.fit(df_na)

    # when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = LogCpTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_raises_error_if_negative_values_after_adding_C(df_vartypes):
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

    # when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = LogCpTransformer(base="e", variables=user_var, C=1)
        transformer.fit(df_vartypes)
        transformer.transform(df_neg)

    assert (
        exceptionmsg
        == "Some variables contain zero or negative values after addingconstant C, "
        + "can't apply log"
    )


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = LogCpTransformer()
        transformer.transform(df_vartypes)
