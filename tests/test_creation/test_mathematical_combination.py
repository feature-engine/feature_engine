import pandas as pd
import pytest

from feature_engine.creation import MathematicalCombination


def test_default_parameters(df_vartypes):
    transformer = MathematicalCombination()

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
            "sum(Age-Marks)": [20.9, 21.8, 19.7, 18.6],
            "prod(Age-Marks)": [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            "mean(Age-Marks)": [10.45, 10.9, 9.85, 9.3],
            "std(Age-Marks)": [
                13.505739520663058,
                14.28355697996826,
                12.94005409571382,
                12.303657992645928,
            ],
            "max(Age-Marks)": [20.0, 21.0, 19.0, 18.0],
            "min(Age-Marks)": [0.9, 0.8, 0.7, 0.6],
        }
    )

    # init params
    assert transformer.variables == ["Age", "Marks"]
    assert transformer.math_operations == ["sum", "prod", "mean", "std", "max", "min"]
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        "sum(Age-Marks)": "sum",
        "prod(Age-Marks)": "prod",
        "mean(Age-Marks)": "mean",
        "std(Age-Marks)": "std",
        "max(Age-Marks)": "max",
        "min(Age-Marks)": "min",
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_input_variables(df_vartypes):
    transformer = MathematicalCombination(variables=["Age", "Marks"])

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
            "sum(Age-Marks)": [20.9, 21.8, 19.7, 18.6],
            "prod(Age-Marks)": [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            "mean(Age-Marks)": [10.45, 10.9, 9.85, 9.3],
            "std(Age-Marks)": [
                13.505739520663058,
                14.28355697996826,
                12.94005409571382,
                12.303657992645928,
            ],
            "max(Age-Marks)": [20.0, 21.0, 19.0, 18.0],
            "min(Age-Marks)": [0.9, 0.8, 0.7, 0.6],
        }
    )

    # init params
    assert transformer.variables == ["Age", "Marks"]
    assert transformer.math_operations == ["sum", "prod", "mean", "std", "max", "min"]
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        "sum(Age-Marks)": "sum",
        "prod(Age-Marks)": "prod",
        "mean(Age-Marks)": "mean",
        "std(Age-Marks)": "std",
        "max(Age-Marks)": "max",
        "min(Age-Marks)": "min",
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_error_when_user_enters_one_variable(df_vartypes):
    with pytest.raises(KeyError):
        MathematicalCombination(variables=["Age"])


def test_error_when_entered_variable_not_numeric(df_vartypes):
    transformer = MathematicalCombination(variables=["Name", "Age", "Marks"])
    with pytest.raises(TypeError):
        transformer.fit_transform(df_vartypes)


def test_error_when_entered_variables_not_in_df(df_vartypes):
    transformer = MathematicalCombination(variables=["FeatOutsideDataset", "Age"])
    with pytest.raises(KeyError):
        transformer.fit_transform(df_vartypes)


def test_user_enters_two_operations(df_vartypes):
    transformer = MathematicalCombination(math_operations=["sum", "mean"])

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
            "sum(Age-Marks)": [20.9, 21.8, 19.7, 18.6],
            "mean(Age-Marks)": [10.45, 10.9, 9.85, 9.3],
        }
    )

    # init params
    assert transformer.variables == ["Age", "Marks"]
    assert transformer.math_operations == ["sum", "mean"]
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        "sum(Age-Marks)": "sum",
        "mean(Age-Marks)": "mean",
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_user_enters_output_variable_names(df_vartypes):
    transformer = MathematicalCombination(
        math_operations=["sum", "mean"],
        new_variables_names=["sum_of_two_vars", "mean_of_two_vars"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
            "sum_of_two_vars": [20.9, 21.8, 19.7, 18.6],
            "mean_of_two_vars": [10.45, 10.9, 9.85, 9.3],
        }
    )

    # init params
    assert transformer.variables == ["Age", "Marks"]
    assert transformer.math_operations == ["sum", "mean"]
    assert transformer.new_variables_names == ["sum_of_two_vars", "mean_of_two_vars"]
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        "sum_of_two_vars": "sum",
        "mean_of_two_vars": "mean",
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_error_if_variable_names_and_operations_list_length_not_equal(df_vartypes):
    with pytest.raises(KeyError):
        MathematicalCombination(
            math_operations=["sum", "mean"],
            new_variables_names=[
                "sum_of_two_vars",
                "mean_of_two_vars",
                "another_alias",
                "not_permitted",
            ],
        )

    with pytest.raises(KeyError):
        MathematicalCombination(
            math_operations=["sum", "mean"], new_variables_names=["sum_of_two_vars"]
        )


def test_one_mathematical_operation(df_vartypes):
    # case 2: selected only one operation:
    transformer = MathematicalCombination(math_operations=["sum"])

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
            "sum(Age-Marks)": [20.9, 21.8, 19.7, 18.6],
        }
    )

    # init params
    assert transformer.variables == ["Age", "Marks"]
    assert transformer.math_operations == ["sum"]
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {"sum(Age-Marks)": "sum"}
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_error_if_operation_not_permitted(df_vartypes):
    with pytest.raises(KeyError):
        MathematicalCombination(math_operations=["OperationOutsidePermittedList"])


def test_error_if_operation_is_wrong_type(df_vartypes):
    with pytest.raises(KeyError):
        MathematicalCombination(math_operations=[sum])


def test_math_operations_type():
    with pytest.raises(KeyError):
        MathematicalCombination(math_operations=("sum", "mean"))
