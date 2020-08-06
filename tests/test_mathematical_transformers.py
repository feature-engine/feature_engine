import pandas as pd
import pytest

from feature_engine.mathematical_transformers import AdditionTransformer


def test_AdditionTransformer_all(dataframe_vartypes):
    transformer = AdditionTransformer()

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum(Age,Marks)': [20.9, 21.8, 19.7, 18.6],
            'prod(Age,Marks)': [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            'mean(Age,Marks)': [10.45, 10.9, 9.85, 9.3],
            'std(Age,Marks)': [13.505739520663058, 14.28355697996826, 12.94005409571382, 12.303657992645928],
            'max(Age,Marks)': [20.0, 21.0, 19.0, 18.0],
            'min(Age,Marks)': [0.9, 0.8, 0.7, 0.6]
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations is None
    assert transformer.operations_ == ['sum', 'prod', 'mean', 'std', 'max', 'min']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_AdditionalTransformer_SelectedVariables(dataframe_vartypes):
    transformer = AdditionTransformer(variables=['Age', 'Marks'])

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum(Age,Marks)': [20.9, 21.8, 19.7, 18.6],
            'prod(Age,Marks)': [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            'mean(Age,Marks)': [10.45, 10.9, 9.85, 9.3],
            'std(Age,Marks)': [13.505739520663058, 14.28355697996826, 12.94005409571382, 12.303657992645928],
            'max(Age,Marks)': [20.0, 21.0, 19.0, 18.0],
            'min(Age,Marks)': [0.9, 0.8, 0.7, 0.6]
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations is None
    assert transformer.operations_ == ['sum', 'prod', 'mean', 'std', 'max', 'min']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_AdditionalTransformer_SelectedVariables_OnlyOneFeature(dataframe_vartypes):
    transformer = AdditionTransformer(variables=['Age'])
    with pytest.raises(KeyError):
        X = transformer.fit_transform(dataframe_vartypes)


def test_AdditionalTransformer_SelectedVariables_NonNumeric(dataframe_vartypes):
    transformer = AdditionTransformer(variables=['Name', 'Age', 'Marks'])
    with pytest.raises(TypeError):
        X = transformer.fit_transform(dataframe_vartypes)


def test_AdditionalTransformer_SelectedVariables_OutsideDataset(dataframe_vartypes):
    transformer = AdditionTransformer(variables=['FeatOutsideDataset'])
    with pytest.raises(KeyError):
        X = transformer.fit_transform(dataframe_vartypes)


def test_AdditionalTransformer_SelectedOperations(dataframe_vartypes):
    transformer = AdditionTransformer(math_operations=['sum', 'mean'])

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum(Age,Marks)': [20.9, 21.8, 19.7, 18.6],
            'mean(Age,Marks)': [10.45, 10.9, 9.85, 9.3],
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations == ['sum', 'mean']
    assert transformer.operations_ == ['sum', 'mean']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_AdditionalTransformer_SelectedOperations_OnlyOneOperation(dataframe_vartypes):
    # case 2: selected only one operation:
    transformer = AdditionTransformer(math_operations=['sum'])

    X = transformer.fit_transform(dataframe_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            'Name': ['tom', 'nick', 'krish', 'jack'],
            'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
            'Age': [20, 21, 19, 18],
            'Marks': [0.9, 0.8, 0.7, 0.6],
            'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
            'sum(Age,Marks)': [20.9, 21.8, 19.7, 18.6]
        }
    )

    # init params
    assert transformer.variables == ['Age', 'Marks']
    assert transformer.math_operations == ['sum']
    assert transformer.operations_ == ['sum']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_AdditionalTransformer_SelectedOperations_OutsidePermittedList(dataframe_vartypes):
    with pytest.raises(KeyError):
        transformer = AdditionTransformer(math_operations=['OperationOutsidePermittedList'])


def test_AdditionalTransformer_SelectedOperations_WrongType(dataframe_vartypes):
    with pytest.raises(KeyError):
        transformer = AdditionTransformer(math_operations=[sum])
