import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_boston

from feature_engine.discretisation import ArbitraryDiscretiser



def test_arbitrary_discretiser():
    boston_dataset = load_boston()
    data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    user_dict = {"LSTAT": [0, 10, 20, 30, np.Inf]}

    data_t1 = data.copy()
    data_t2 = data.copy()
    data_t1["LSTAT"] = pd.cut(data["LSTAT"], bins=[0, 10, 20, 30, np.Inf])
    data_t2["LSTAT"] = pd.cut(data["LSTAT"], bins=[0, 10, 20, 30, np.Inf], labels=False)

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_object=False, return_boundaries=False
    )
    X = transformer.fit_transform(data)

    # init params
    assert transformer.return_object is False
    assert transformer.return_boundaries is False
    # fit params
    assert transformer.variables_ == ["LSTAT"]
    assert transformer.binner_dict_ == user_dict
    # transform params
    pd.testing.assert_frame_equal(X, data_t2)

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_object=False, return_boundaries=True
    )
    X = transformer.fit_transform(data)
    pd.testing.assert_frame_equal(X, data_t1)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method

    msg = "During the discretisation, NaN values were introduced " \
          "in the feature(s) var_A."

    # check for warning when errors equals 'ignore'
    with pytest.warns(UserWarning) as record:
        transformer = ArbitraryDiscretiser(errors="ignore")
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])

    # check that only one warning was returned
    assert len(record) == 1
    # check that message matches
    assert record[0].message.args[0] == msg

    # check for error when errors equals 'raise'
    with pytest.raises(ValueError) as record:
        transformer = ArbitraryDiscretiser(errors="raise")
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])

    # check that error message matches
    assert str(record.value) == msg


def test_error_if_not_permitted_value_is_errors():
    with pytest.raises(ValueError):
        ArbitraryDiscretiser(errors="medialuna")
