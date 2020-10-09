import pandas as pd
import pytest

from feature_engine.creation import MathematicalCombination


def test_math_combination_default_parameters(dataframe_vartypes):
    transformer = MathematicalCombination()

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum(Age-Marks)': [20.9, 21.8, 19.7, 18.6],
            'prod(Age-Marks)': [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            'mean(Age-Marks)': [10.45, 10.9, 9.85, 9.3],
            'std(Age-Marks)': [13.505739520663058, 14.28355697996826, 12.94005409571382, 12.303657992645928],
            'max(Age-Marks)': [20.0, 21.0, 19.0, 18.0],
            'min(Age-Marks)': [0.9, 0.8, 0.7, 0.6]
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations == ['sum', 'prod', 'mean', 'std', 'max', 'min']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        'sum(Age-Marks)': 'sum',
        'prod(Age-Marks)': 'prod',
        'mean(Age-Marks)': 'mean',
        'std(Age-Marks)': 'std',
        'max(Age-Marks)': 'max',
        'min(Age-Marks)': 'min'
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_math_combination_select_variables(dataframe_vartypes):
    transformer = MathematicalCombination(variables=['Age', 'Marks'])

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum(Age-Marks)': [20.9, 21.8, 19.7, 18.6],
            'prod(Age-Marks)': [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            'mean(Age-Marks)': [10.45, 10.9, 9.85, 9.3],
            'std(Age-Marks)': [13.505739520663058, 14.28355697996826, 12.94005409571382, 12.303657992645928],
            'max(Age-Marks)': [20.0, 21.0, 19.0, 18.0],
            'min(Age-Marks)': [0.9, 0.8, 0.7, 0.6]
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations == ['sum', 'prod', 'mean', 'std', 'max', 'min']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        'sum(Age-Marks)': 'sum',
        'prod(Age-Marks)': 'prod',
        'mean(Age-Marks)': 'mean',
        'std(Age-Marks)': 'std',
        'max(Age-Marks)': 'max',
        'min(Age-Marks)': 'min'
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_math_combination_error_when_user_selects_one_variable(dataframe_vartypes):
    with pytest.raises(KeyError):
        transformer = MathematicalCombination(variables=['Age'])


def test_math_combination_error_when_selected_variable_not_numeric(dataframe_vartypes):
    transformer = MathematicalCombination(variables=['Name', 'Age', 'Marks'])
    with pytest.raises(TypeError):
        X = transformer.fit_transform(dataframe_vartypes)


def test_math_combination_error_when_selected_variable_not_in_df(dataframe_vartypes):
    transformer = MathematicalCombination(variables=['FeatOutsideDataset', 'Age'])
    with pytest.raises(KeyError):
        X = transformer.fit_transform(dataframe_vartypes)


def test_math_combination_select_two_operations(dataframe_vartypes):
    transformer = MathematicalCombination(math_operations=['sum', 'mean'])

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum(Age-Marks)': [20.9, 21.8, 19.7, 18.6],
            'mean(Age-Marks)': [10.45, 10.9, 9.85, 9.3],
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations == ['sum', 'mean']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        'sum(Age-Marks)': 'sum',
        'mean(Age-Marks)': 'mean'
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_math_combination_user_enters_new_variable_names(dataframe_vartypes):
    transformer = MathematicalCombination(
        math_operations=['sum', 'mean'],
        new_variables_names=['sum_of_two_vars', 'mean_of_two_vars']
    )

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum_of_two_vars': [20.9, 21.8, 19.7, 18.6],
            'mean_of_two_vars': [10.45, 10.9, 9.85, 9.3],
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations == ['sum', 'mean']
    assert transformer.new_variables_names == ['sum_of_two_vars', 'mean_of_two_vars']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        'sum_of_two_vars': 'sum',
        'mean_of_two_vars': 'mean'
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_math_combination_error_when_variable_names_and_operations_list_length_not_equal(dataframe_vartypes):
    with pytest.raises(KeyError):
        transformer = MathematicalCombination(
            math_operations=['sum', 'mean'],
            new_variables_names=['sum_of_two_vars', 'mean_of_two_vars', 'another_alias', 'not_permitted']
        )

    with pytest.raises(KeyError):
        transformer = MathematicalCombination(
            math_operations=['sum', 'mean'],
            new_variables_names=['sum_of_two_vars']
        )

def test_math_combination_only_one_mathematical_operation(dataframe_vartypes):
    # case 2: selected only one operation:
    transformer = MathematicalCombination(math_operations=['sum'])

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum(Age-Marks)': [20.9, 21.8, 19.7, 18.6]
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations == ['sum']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    assert transformer.combination_dict_ == {
        'sum(Age-Marks)': 'sum'
    }
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_math_combination_selected_operation_not_in_permitted_list(dataframe_vartypes):
    with pytest.raises(KeyError):
        transformer = MathematicalCombination(math_operations=['OperationOutsidePermittedList'])


def test_math_combination_selected_operation_is_wrong_type(dataframe_vartypes):
    with pytest.raises(KeyError):
        transformer = MathematicalCombination(math_operations=[sum])
