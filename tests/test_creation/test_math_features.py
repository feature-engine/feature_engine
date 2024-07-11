import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.creation import MathFeatures

dob_datrange = pd.date_range("2020-02-24", periods=4, freq="min")


# test param variables_to_combine
def test_error_when_required_params_not_entered():
    with pytest.raises(TypeError):
        MathFeatures(func=["mean", "sum"])
    with pytest.raises(TypeError):
        MathFeatures(variables=["vara", "varb"])


@pytest.mark.parametrize(
    "_variables",
    [
        ["vara", "vara", "varb"],
        ["vara", "vara"],
        "vara",
        ["vara"],
        1,
        [1, 1, 2],
        [2],
        ["vara", 0.5],
    ],
)
def test_error_when_variables_not_permitted(_variables):
    with pytest.raises(ValueError):
        MathFeatures(variables=_variables, func=["sum", "mean"])


def test_error_if_func_is_dictionary():
    with pytest.raises(NotImplementedError):
        MathFeatures(variables=["Age", "Name"], func={"A": "sum", "B": "mean"})


@pytest.mark.parametrize("_variables", [[4], ("vara", "vara"), "vara"])
def test_error_if_new_variable_names_not_permitted(_variables):
    with pytest.raises(ValueError):
        MathFeatures(
            variables=["Age", "Name"], func=["sum"], new_variables_names=_variables
        )


def test_error_new_variable_names_not_permitted():
    variables = ["Age", "Name"]
    with pytest.raises(ValueError):
        MathFeatures(
            variables=variables,
            func=["sum", "mean"],
            new_variables_names=[
                "sum_of_two_vars",
                "mean_of_two_vars",
                "another_alias",
            ],
        )

    with pytest.raises(ValueError):
        MathFeatures(
            variables=variables,
            func=["sum"],
            new_variables_names=["sum_of_two_vars", "mean_of_two_vars"],
        )

    with pytest.raises(ValueError):
        MathFeatures(
            variables=variables,
            func="sum",
            new_variables_names=["sum_of_two_vars", "mean_of_two_vars"],
        )
    with pytest.raises(ValueError):
        MathFeatures(
            variables=variables,
            func=["sum", "mean"],
            new_variables_names=["sum_of_two_vars", "sum_of_two_vars"],
        )


def test_aggregations_with_strings(df_vartypes):
    transformer = MathFeatures(
        variables=["Age", "Marks"], func=["sum", "prod", "mean", "std", "max", "min"]
    )
    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": dob_datrange,
            "sum_Age_Marks": [20.9, 21.8, 19.7, 18.6],
            "prod_Age_Marks": [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            "mean_Age_Marks": [10.45, 10.9, 9.85, 9.3],
            "std_Age_Marks": [
                13.505739520663058,
                14.28355697996826,
                12.94005409571382,
                12.303657992645928,
            ],
            "max_Age_Marks": [20.0, 21.0, 19.0, 18.0],
            "min_Age_Marks": [0.9, 0.8, 0.7, 0.6],
        }
    )

    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_aggregations_with_functions(df_vartypes):
    transformer = MathFeatures(
        variables=["Age", "Marks"], func=[np.sum, np.mean, np.std]
    )
    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": dob_datrange,
            "sum_Age_Marks": [20.9, 21.8, 19.7, 18.6],
            "mean_Age_Marks": [10.45, 10.9, 9.85, 9.3],
            "std_Age_Marks": [
                13.505739520663058,
                14.28355697996826,
                12.94005409571382,
                12.303657992645928,
            ],
        }
    )

    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_user_enters_two_operations(df_vartypes):
    transformer = MathFeatures(variables=["Age", "Marks"], func=["sum", np.mean])

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": dob_datrange,
            "sum_Age_Marks": [20.9, 21.8, 19.7, 18.6],
            "mean_Age_Marks": [10.45, 10.9, 9.85, 9.3],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


def test_new_variable_names(df_vartypes):
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        new_variables_names=["sum_of_two_vars", "mean_of_two_vars"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": dob_datrange,
            "sum_of_two_vars": [20.9, 21.8, 19.7, 18.6],
            "mean_of_two_vars": [10.45, 10.9, 9.85, 9.3],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


def test_one_mathematical_operation(df_vartypes):
    transformer = MathFeatures(variables=["Age", "Marks"], func="sum")
    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": dob_datrange,
            "sum_Age_Marks": [20.9, 21.8, 19.7, 18.6],
        }
    )
    pd.testing.assert_frame_equal(X, ref)

    transformer = MathFeatures(variables=["Age", "Marks"], func=["sum"])
    X = transformer.fit_transform(df_vartypes)
    pd.testing.assert_frame_equal(X, ref)


def test_variable_names_when_df_cols_are_integers(df_numeric_columns):
    transformer = MathFeatures(
        variables=[2, 3], func=["sum", "prod", "mean", "std", "max", "min"]
    )

    X = transformer.fit_transform(df_numeric_columns)

    ref = pd.DataFrame.from_dict(
        {
            0: ["tom", "nick", "krish", "jack"],
            1: ["London", "Manchester", "Liverpool", "Bristol"],
            2: [20, 21, 19, 18],
            3: [0.9, 0.8, 0.7, 0.6],
            4: dob_datrange,
            "sum_2_3": [20.9, 21.8, 19.7, 18.6],
            "prod_2_3": [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            "mean_2_3": [10.45, 10.9, 9.85, 9.3],
            "std_2_3": [
                13.505739520663058,
                14.28355697996826,
                12.94005409571382,
                12.303657992645928,
            ],
            "max_2_3": [20.0, 21.0, 19.0, 18.0],
            "min_2_3": [0.9, 0.8, 0.7, 0.6],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


def test_error_when_null_values_in_variable(df_vartypes):

    df_na = df_vartypes.copy()
    df_na.loc[1, "Age"] = np.nan

    math_combinator = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        missing_values="raise",
    )

    with pytest.raises(ValueError):
        math_combinator.fit(df_na)

    math_combinator.fit(df_vartypes)
    with pytest.raises(ValueError):
        math_combinator.transform(df_na)


def test_no_error_when_null_values_in_variable(df_vartypes):

    df_na = df_vartypes.copy()
    df_na.loc[1, "Age"] = np.nan

    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        missing_values="ignore",
    )

    X = transformer.fit_transform(df_na)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, np.nan, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": dob_datrange,
            "sum_Age_Marks": [20.9, 0.8, 19.7, 18.6],
            "mean_Age_Marks": [10.45, 0.8, 9.85, 9.3],
        }
    )
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_drop_original_variables(df_vartypes):
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        drop_original=True,
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "dob": dob_datrange,
            "sum_Age_Marks": [20.9, 21.8, 19.7, 18.6],
            "mean_Age_Marks": [10.45, 10.9, 9.85, 9.3],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


@pytest.mark.parametrize("_varnames", [None, ["var1", "var2"]])
@pytest.mark.parametrize("_drop", [True, False])
def test_get_feature_names_out(_varnames, _drop, df_vartypes):
    tr = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        new_variables_names=_varnames,
        drop_original=_drop,
    )
    X = tr.fit_transform(df_vartypes)
    feat_out = list(X.columns)
    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=df_vartypes.columns) == feat_out


@pytest.mark.parametrize("_varnames", [None, ["var1", "var2"]])
@pytest.mark.parametrize("_drop", [True, False])
def test_get_feature_names_out_from_pipeline(_varnames, _drop, df_vartypes):

    # set up transformer
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        new_variables_names=_varnames,
        drop_original=_drop,
    )

    pipe = Pipeline([("transformer", transformer)])

    # fit transformer
    X = pipe.fit_transform(df_vartypes)

    feat_out = list(X.columns)
    assert pipe.get_feature_names_out(input_features=None) == feat_out
    assert pipe.get_feature_names_out(input_features=df_vartypes.columns) == feat_out


@pytest.mark.parametrize("_input_features", ["hola", ["Age", "Marks"]])
def test_get_feature_names_out_raises_error_when_wrong_param(
    _input_features, df_vartypes
):
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
    )
    transformer.fit(df_vartypes)

    with pytest.raises(ValueError):
        transformer.get_feature_names_out(input_features=_input_features)
