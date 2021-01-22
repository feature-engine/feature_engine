import pandas as pd
import pytest

from feature_engine.creation import CombineWithReferenceFeature


# test param variables_to_combine
def test_error_when_param_variables_not_entered():
    with pytest.raises(TypeError):
        CombineWithReferenceFeature()


def test_error_when_variables_to_combine_wrong_type():
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=("Age", "Name"), reference_variables=["Age", "Name"]
        )
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=["Age", 0, 4.5], reference_variables=["Age", "Name"]
        )
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=["Age", "Name"], reference_variables=("Age", "Name")
        )
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=["Age", "Name"], reference_variables=["Age", 0, 4.5]
        )


# test param operations
def test_error_if_operation_not_supported():
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=["Age", "Name"],
            reference_variables=["Age", "Name"],
            operations=["an_operation"],
        )


def test_error_if_operation_is_wrong_type():
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=["Age", "Name"],
            reference_variables=["Age", "Name"],
            operations=[sum],
        )
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=["Age", "Name"],
            reference_variables=["Age", "Name"],
            operations=("sub", "div"),
        )


# test new variable names
def test_error_if_new_variable_names_of_wrong_type():
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=["Age", "Name"],
            reference_variables=["Age", "Name"],
            new_variables_names=[4],
        )
    with pytest.raises(ValueError):
        CombineWithReferenceFeature(
            variables_to_combine=["Age", "Name"],
            reference_variables=["Age", "Name"],
            new_variables_names=("var1", "var2"),
        )


def test_error_when_variables_to_combine_not_numeric(df_vartypes):
    transformer = CombineWithReferenceFeature(
        variables_to_combine=["Name", "Age", "Marks"],
        reference_variables=["Age", "Name"],
        operations=["sub"],
    )
    with pytest.raises(TypeError):
        transformer.fit_transform(df_vartypes)


def test_error_when_entered_variables_not_in_df(df_vartypes):
    transformer = CombineWithReferenceFeature(
        variables_to_combine=["FeatOutsideDataset", "Age"],
        reference_variables=["Age", "Name"],
        operations=["sub"],
    )
    with pytest.raises(KeyError):
        transformer.fit_transform(df_vartypes)


def test_all_binary_operation(df_vartypes):
    # case 2: selected only one operation:
    transformer = CombineWithReferenceFeature(
        variables_to_combine=["Age"],
        reference_variables=["Marks"],
        operations=["sub", "div", "add", "mul"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
            "Age_sub_Marks": [19.1, 20.2, 18.3, 17.4],
            "Age_div_Marks": [22.22222222222222, 26.25, 27.142857142857146, 30.0],
            "Age_add_Marks": [20.9, 21.8, 19.7, 18.6],
            "Age_mul_Marks": [18.0, 16.8, 13.299999999999999, 10.799999999999999],
        }
    )

    # init params
    assert transformer.variables_to_combine == ["Age"]
    assert transformer.reference_variables == ["Marks"]
    assert transformer.operations == ["sub", "div", "add", "mul"]
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_operations_with_multiple_variables(df_vartypes):
    transformer = CombineWithReferenceFeature(
        variables_to_combine=["Age", "Marks"],
        reference_variables=["Age", "Marks"],
        operations=["sub"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
            "Age_sub_Age": [0, 0, 0, 0],
            "Marks_sub_Age": [-19.1, -20.2, -18.3, -17.4],
            "Age_sub_Marks": [19.1, 20.2, 18.3, 17.4],
            "Marks_sub_Marks": [0.0, 0.0, 0.0, 0.0],
        }
    )

    # init params
    assert transformer.variables_to_combine == ["Age", "Marks"]
    assert transformer.reference_variables == ["Age", "Marks"]
    # fit params
    assert transformer.operations == ["sub"]
    assert transformer.input_shape_ == (4, 5)

    # transform params
    pd.testing.assert_frame_equal(X, ref)


def test_user_enters_output_variable_names(df_vartypes):
    transformer = CombineWithReferenceFeature(
        variables_to_combine=["Age", "Marks"],
        reference_variables=["Age", "Marks"],
        operations=["sub"],
        new_variables_names=["Juan", "Pedro", "Fasola", "GranAmigo"],
    )

    X = transformer.fit_transform(df_vartypes)

    ref = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
            "Juan": [0, 0, 0, 0],
            "Pedro": [-19.1, -20.2, -18.3, -17.4],
            "Fasola": [19.1, 20.2, 18.3, 17.4],
            "GranAmigo": [0.0, 0.0, 0.0, 0.0],
        }
    )

    # init params
    assert transformer.variables_to_combine == ["Age", "Marks"]
    assert transformer.reference_variables == ["Age", "Marks"]
    # fit params
    assert transformer.operations == ["sub"]
    assert transformer.input_shape_ == (4, 5)

    # transform params
    pd.testing.assert_frame_equal(X, ref)
