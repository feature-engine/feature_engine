import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import PowerTransformer


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