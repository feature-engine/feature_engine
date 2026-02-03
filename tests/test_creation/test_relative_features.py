import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.creation import RelativeFeatures


def test_mandatory_init_parameters():
    with pytest.raises(TypeError):
        RelativeFeatures(reference=["var1"], func=["add"])
    with pytest.raises(TypeError):
        RelativeFeatures(variables=["var1"], func=["add"])
    with pytest.raises(TypeError):
        RelativeFeatures(variables=["var1"], reference=["var2"])


_variables = ["var1", ["var1", "var1", "var2"], ["var1", 0.5], ("Age", "Name")]


@pytest.mark.parametrize("_variables", _variables)
def test_error_when_param_variables_not_permitted(_variables):
    with pytest.raises(ValueError):
        RelativeFeatures(
            variables=_variables, reference=["Age", "Name"], func=["add", "mul"]
        )


@pytest.mark.parametrize("_variables", _variables)
def test_error_when_param_reference_not_permitted(_variables):
    with pytest.raises(ValueError):
        RelativeFeatures(
            reference=_variables, variables=["Age", "Name"], func=["add", "mul"]
        )


_operations = [
    "add",
    ["add", "add", "mul"],
    ["add", "multiply"],
    ("add", "mul"),
    [np.mean, "add"],
]


@pytest.mark.parametrize("_func", _operations)
def test_error_if_func_not_supported(_func):
    with pytest.raises(ValueError):
        RelativeFeatures(
            variables=["Age", "Name"],
            reference=["Age", "Name"],
            func=_func,
        )


@pytest.mark.parametrize("_fill_value", [(2, 3.3), ["test"], "python"])
def test_error_if_fill_value_not_permitted(_fill_value):
    with pytest.raises(ValueError):
        RelativeFeatures(
            variables=["Age"],
            reference=["Marks"],
            func=["sub", "div", "add", "mul"],
            fill_value=_fill_value,
        )


def test_error_when_drop_original_not_bool():
    for drop_original in ["True", [True]]:
        with pytest.raises(ValueError):
            RelativeFeatures(
                variables=["Age"],
                reference=["Marks"],
                func=["add", "mul"],
                drop_original=drop_original,
            )


def test_error_when_variables_not_numeric(df_vartypes):
    transformer = RelativeFeatures(
        variables=["Name", "Age", "Marks"],
        reference=["Age", "Name"],
        func=["sub"],
    )
    with pytest.raises(TypeError):
        transformer.fit_transform(df_vartypes)

    transformer = RelativeFeatures(
        reference=["Name", "Age", "Marks"],
        variables=["Age", "Name"],
        func=["sub"],
    )
    with pytest.raises(TypeError):
        transformer.fit_transform(df_vartypes)


def test_error_when_entered_variables_not_in_df(df_vartypes):
    transformer = RelativeFeatures(
        variables=["FeatOutsideDataset", "Age"],
        reference=["Age", "Name"],
        func=["sub"],
    )
    with pytest.raises(KeyError):
        transformer.fit_transform(df_vartypes)

    transformer = RelativeFeatures(
        reference=["FeatOutsideDataset", "Age"],
        variables=["Age", "Name"],
        func=["sub"],
    )
    with pytest.raises(TypeError):
        transformer.fit_transform(df_vartypes)


def test_classic_binary_operation(df_vartypes):

    transformer = RelativeFeatures(
        variables=["Age"],
        reference=["Marks"],
        func=["sub", "div", "add", "mul"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
            "Age_sub_Marks": [19.1, 20.2, 18.3, 17.4],
            "Age_div_Marks": [22.22222222222222, 26.25, 27.142857142857146, 30.0],
            "Age_add_Marks": [20.9, 21.8, 19.7, 18.6],
            "Age_mul_Marks": [18.0, 16.8, 13.299999999999999, 10.799999999999999],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


def test_alternative_operation(df_vartypes):

    # input df
    df = df_vartypes.copy()

    # Expected result
    dft = df.copy()
    dft["Age_truediv_Marks"] = dft["Age"].truediv(dft["Marks"])
    dft["Age_floordiv_Marks"] = dft["Age"].floordiv(dft["Marks"])
    dft["Age_mod_Marks"] = dft["Age"].mod(dft["Marks"])
    dft["Age_pow_Marks"] = dft["Age"].pow(dft["Marks"])

    transformer = RelativeFeatures(
        variables=["Age"],
        reference=["Marks"],
        func=["truediv", "floordiv", "mod", "pow"],
    )
    X = transformer.fit_transform(df)

    pd.testing.assert_frame_equal(X, dft)


def test_operations_with_multiple_variables(df_vartypes):
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["sub"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
            "Age_sub_Age": [0, 0, 0, 0],
            "Marks_sub_Age": [-19.1, -20.2, -18.3, -17.4],
            "Age_sub_Marks": [19.1, 20.2, 18.3, 17.4],
            "Marks_sub_Marks": [0.0, 0.0, 0.0, 0.0],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


def test_multiple_operations_with_multiple_variables(df_vartypes):
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["sub", "add"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
            "Age_sub_Age": [0, 0, 0, 0],
            "Marks_sub_Age": [-19.1, -20.2, -18.3, -17.4],
            "Age_sub_Marks": [19.1, 20.2, 18.3, 17.4],
            "Marks_sub_Marks": [0.0, 0.0, 0.0, 0.0],
            "Age_add_Age": [40, 42, 38, 36],
            "Marks_add_Age": [20.9, 21.8, 19.7, 18.6],
            "Age_add_Marks": [20.9, 21.8, 19.7, 18.6],
            "Marks_add_Marks": [1.8, 1.6, 1.4, 1.2],
        }
    )

    pd.testing.assert_frame_equal(X, ref)

    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "sub"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
            "Age_add_Age": [40, 42, 38, 36],
            "Marks_add_Age": [20.9, 21.8, 19.7, 18.6],
            "Age_add_Marks": [20.9, 21.8, 19.7, 18.6],
            "Marks_add_Marks": [1.8, 1.6, 1.4, 1.2],
            "Age_sub_Age": [0, 0, 0, 0],
            "Marks_sub_Age": [-19.1, -20.2, -18.3, -17.4],
            "Age_sub_Marks": [19.1, 20.2, 18.3, 17.4],
            "Marks_sub_Marks": [0.0, 0.0, 0.0, 0.0],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


def test_when_missing_values_is_ignore(df_vartypes):

    df_na = df_vartypes.copy()
    df_na.loc[1, "Age"] = np.nan

    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["sub"],
        missing_values="ignore",
    )

    X = transformer.fit_transform(df_na)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, np.nan, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
            "Age_sub_Age": [0, np.nan, 0, 0],
            "Marks_sub_Age": [-19.1, np.nan, -18.3, -17.4],
            "Age_sub_Marks": [19.1, np.nan, 18.3, 17.4],
            "Marks_sub_Marks": [0.0, 0.0, 0.0, 0.0],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


def test_error_when_null_values_in_variable(df_vartypes):

    df_na = df_vartypes.copy()
    df_na.loc[1, "Age"] = np.nan

    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "mul"],
        missing_values="raise",
    )

    with pytest.raises(ValueError):
        transformer.fit(df_na)

    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na)


def test_when_df_cols_are_integers(df_vartypes):
    df = df_vartypes.copy()
    df.columns = [0, 1, 2, 3, 4]

    transformer = RelativeFeatures(
        variables=[2, 3],
        reference=[2, 3],
        func=["sub", "add"],
    )

    X = transformer.fit_transform(df)

    ref = pd.DataFrame.from_dict(
        {
            0: ["tom", "nick", "krish", "jack"],
            1: ["London", "Manchester", "Liverpool", "Bristol"],
            2: [20, 21, 19, 18],
            3: [0.9, 0.8, 0.7, 0.6],
            4: pd.date_range("2020-02-24", periods=4, freq="min"),
            "2_sub_2": [0, 0, 0, 0],
            "3_sub_2": [-19.1, -20.2, -18.3, -17.4],
            "2_sub_3": [19.1, 20.2, 18.3, 17.4],
            "3_sub_3": [0.0, 0.0, 0.0, 0.0],
            "2_add_2": [40, 42, 38, 36],
            "3_add_2": [20.9, 21.8, 19.7, 18.6],
            "2_add_3": [20.9, 21.8, 19.7, 18.6],
            "3_add_3": [1.8, 1.6, 1.4, 1.2],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


@pytest.mark.parametrize("_func", [["div"], ["truediv"], ["floordiv"], ["mod"]])
def test_error_when_division_by_zero_and_fill_value_is_none(_func, df_vartypes):

    df_zero = df_vartypes.copy()
    df_zero.loc[1, "Marks"] = 0

    transformer = RelativeFeatures(
        variables=["Age"],
        reference=["Marks"],
        func=_func,
    )
    transformer.fit(df_vartypes)

    with pytest.raises(ValueError) as record:
        transformer.transform(df_zero)

    msg = (
        "Some of the reference variables contain zeroes. Division by zero "
        "does not exist. Replace zeros before using this transformer for division "
        "or set `fill_value` to a number."
    )
    # check that the error message matches
    assert str(record.value) == msg


@pytest.mark.parametrize(
    "_fill_value, _func",
    [
        (111.111, ["div"]),
        (999, ["div"]),
        (111.111, ["truediv"]),
        (999, ["truediv"]),
        (111.111, ["floordiv"]),
        (999, ["floordiv"]),
        (111.111, ["mod"]),
        (999, ["mod"]),
    ],
)
def test_fill_values_when_division_by_zero(_fill_value, _func, df_vartypes):
    df_zero = df_vartypes.copy()
    df_zero.loc[2, "Marks"] = 0
    df_zero.loc[1, "Age"] = np.nan
    df_zero.loc[3, "Age"] = np.inf

    transformer = RelativeFeatures(
        variables=["Age"],
        reference=["Marks"],
        fill_value=_fill_value,
        func=_func,
        missing_values="ignore",
    )

    X = transformer.fit_transform(df_zero)

    new_var = f"Age_{_func[0]}_Marks"

    assert X.loc[2, new_var] == _fill_value
    np.testing.assert_equal(X.loc[1, "Age"], np.nan)
    np.testing.assert_equal(X.loc[3, "Age"], np.inf)


@pytest.mark.parametrize("_drop", [True, False])
def test_get_feature_names_out(_drop, df_vartypes):
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "sub"],
        drop_original=_drop,
    )
    varnames = [
        "Age_add_Age",
        "Marks_add_Age",
        "Age_add_Marks",
        "Marks_add_Marks",
        "Age_sub_Age",
        "Marks_sub_Age",
        "Age_sub_Marks",
        "Marks_sub_Marks",
    ]
    X = transformer.fit_transform(df_vartypes)
    feat_out = list(X.columns)
    assert feat_out == transformer.get_feature_names_out(input_features=None)
    assert feat_out == transformer.get_feature_names_out(
        input_features=df_vartypes.columns
    )
    assert all([f for f in varnames if f in feat_out])


@pytest.mark.parametrize("_drop", [True, False])
def test_get_feature_names_out_from_pipeline(_drop, df_vartypes):
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "sub"],
        drop_original=_drop,
    )

    pipe = Pipeline([("transformer", transformer)])

    varnames = [
        "Age_add_Age",
        "Marks_add_Age",
        "Age_add_Marks",
        "Marks_add_Marks",
        "Age_sub_Age",
        "Marks_sub_Age",
        "Age_sub_Marks",
        "Marks_sub_Marks",
    ]

    X = pipe.fit_transform(df_vartypes)
    assert list(X.columns) == pipe.get_feature_names_out(input_features=None)
    assert list(X.columns) == pipe.get_feature_names_out(
        input_features=df_vartypes.columns
    )
    assert all([f for f in varnames if f in X.columns])


@pytest.mark.parametrize("_input_features", ["hola", ["Age", "Marks"]])
def test_get_feature_names_out_raises_error_when_wrong_param(
    _input_features, df_vartypes
):
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "sub"],
    )
    transformer.fit(df_vartypes)

    with pytest.raises(ValueError):
        transformer.get_feature_names_out(input_features=_input_features)
