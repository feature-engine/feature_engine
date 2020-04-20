import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError

from feature_engine.variable_transformers import LogTransformer, ReciprocalTransformer, PowerTransformer
from feature_engine.variable_transformers import BoxCoxTransformer, YeoJohnsonTransformer


def test_LogTransformer(dataframe_vartypes, dataframe_na):
    # test case 1: log base e, automatically select variables
    transformer = LogTransformer(base='e', variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [2.99573, 3.04452, 2.94444, 2.89037]
    transf_df['Marks'] = [-0.105361, -0.223144, -0.356675, -0.510826]

    # init params
    assert transformer.base == 'e'
    assert transformer.variables == ['Age', 'Marks']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    # test case 2: log base 10, user passes variables
    transformer = LogTransformer(base='10', variables='Age')
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [1.30103, 1.32222, 1.27875, 1.25527]

    # init params
    assert transformer.base == '10'
    assert transformer.variables == ['Age']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    with pytest.raises(ValueError):
        LogTransformer(base='other')

    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(dataframe_na)

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    # test error when data contains negative values
    df_neg = dataframe_vartypes.copy()
    df_neg.loc[1, 'Age'] = -1

    # test case 5: when variable contains negative value, fit
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(df_neg)

    # test case 6: when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(df_neg)

    with pytest.raises(NotFittedError):
        transformer = LogTransformer()
        transformer.transform(dataframe_vartypes)


def test_ReciprocalTransformer(dataframe_vartypes, dataframe_na):
    # test case 1: automatically select variables
    transformer = ReciprocalTransformer(variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [0.05, 0.047619, 0.0526316, 0.0555556]
    transf_df['Marks'] = [1.11111, 1.25, 1.42857, 1.66667]

    # init params
    assert transformer.variables == ['Age', 'Marks']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(dataframe_na)

    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    # test error when data contains value zero
    df_neg = dataframe_vartypes.copy()
    df_neg.loc[1, 'Age'] = 0

    # test case 4: when variable contains zero, fit
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(df_neg)

    # test case 5: when variable contains zero, transform
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(df_neg)

    with pytest.raises(NotFittedError):
        transformer = ReciprocalTransformer()
        transformer.transform(dataframe_vartypes)


def test_PowerTransformer(dataframe_vartypes, dataframe_na):
    # test case 1: automatically select variables
    transformer = PowerTransformer(variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [4.47214, 4.58258, 4.3589, 4.24264]
    transf_df['Marks'] = [0.948683, 0.894427, 0.83666, 0.774597]

    # init params
    assert transformer.exp == 0.5
    assert transformer.variables == ['Age', 'Marks']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    with pytest.raises(ValueError):
        PowerTransformer(exp='other')

    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = PowerTransformer()
        transformer.fit(dataframe_na)

    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = PowerTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    with pytest.raises(NotFittedError):
        transformer = PowerTransformer()
        transformer.transform(dataframe_vartypes)


def test_BoxCoxTransformer(dataframe_vartypes, dataframe_na):
    # test case 1: automatically select variables
    transformer = BoxCoxTransformer(variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [9.78731, 10.1666, 9.40189, 9.0099]
    transf_df['Marks'] = [-0.101687, -0.207092, -0.316843, -0.431788]

    # init params
    assert transformer.variables == ['Age', 'Marks']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = BoxCoxTransformer()
        transformer.fit(dataframe_na)

    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = BoxCoxTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    # test error when data contains negative values
    df_neg = dataframe_vartypes.copy()
    df_neg.loc[1, 'Age'] = -1

    # test case 4: when variable contains negative value, fit
    with pytest.raises(ValueError):
        transformer = BoxCoxTransformer()
        transformer.fit(df_neg)

    # test case 5: when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = BoxCoxTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(df_neg)

    with pytest.raises(NotFittedError):
        transformer = BoxCoxTransformer()
        transformer.transform(dataframe_vartypes)


def test_YeoJohnsonTransformer(dataframe_vartypes, dataframe_na):
    # test case 1: automatically select variables
    transformer = YeoJohnsonTransformer(variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [10.167, 10.5406, 9.78774, 9.40229]
    transf_df['Marks'] = [0.804449, 0.722367, 0.638807, 0.553652]

    # init params
    assert transformer.variables == ['Age', 'Marks']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = YeoJohnsonTransformer()
        transformer.fit(dataframe_na)

    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = YeoJohnsonTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    with pytest.raises(NotFittedError):
        transformer = YeoJohnsonTransformer()
        transformer.transform(dataframe_vartypes)
