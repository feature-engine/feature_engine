import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from feature_engine.discretisation import ArbitraryDiscretiser



def test_arbitrary_discretiser():
    california_dataset = fetch_california_housing()
    data = pd.DataFrame(california_dataset.data,
                        columns=california_dataset.feature_names)
    user_dict = {"HouseAge": [0, 20, 40, 60, np.Inf]}

    data_t1 = data.copy()
    data_t2 = data.copy()
    # HouseAge is the median house age in the block group.
    data_t1["HouseAge"] = pd.cut(data["HouseAge"],
                                 bins=[0, 20, 40, 60, np.Inf])
    data_t2["HouseAge"] = pd.cut(data["HouseAge"],
                                 bins=[0, 20, 40, 60, np.Inf],
                                 labels=False)

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_object=False, return_boundaries=False
    )
    X = transformer.fit_transform(data)

    # init params
    assert transformer.return_object is False
    assert transformer.return_boundaries is False
    # fit params
    assert transformer.variables_ == ["HouseAge"]
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
    age_dict = {"Age": [0, 10, 20, 30, np.Inf]}

    with pytest.raises(ValueError):
        transformer = ArbitraryDiscretiser(binning_dict=age_dict)
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_when_nan_introduced_during_transform(df_vartypes, df_na):
    # test case 5: when NA is introduced by the transformation
    msg = "During the discretisation, NaN values were introduced " \
          "in the feature(s) var_A."
    age_dict = {"Age": [0, 10, 20, 30, np.Inf]}

    # check for warning when errors equals 'ignore'
    with pytest.warns(UserWarning) as record:
        transformer = ArbitraryDiscretiser(
            binning_dict=age_dict,
            errors="ignore"
        )
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])

    # check that only one warning was returned
    assert len(record) == 1
    # check that message matches
    assert record[0].message.args[0] == msg

    # check for error when errors equals 'raise'
    with pytest.raises(ValueError) as record:
        transformer = ArbitraryDiscretiser(
            binning_dict=age_dict,
            errors="raise"
        )
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])

    # check that error message matches
    assert str(record.value) == msg


def test_error_if_not_permitted_value_is_errors():
    age_dict = {"Age": [0, 10, 20, 30, np.Inf]}
    with pytest.raises(ValueError):
        ArbitraryDiscretiser(binning_dict=age_dict, errors="medialuna")
