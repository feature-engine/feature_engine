import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from scipy.stats import skewnorm
from sklearn.datasets import fetch_california_housing

from feature_engine.discretisation import ArbitraryDiscretiser


def test_arbitrary_discretiser():
    california_dataset = fetch_california_housing()
    data = pd.DataFrame(
        california_dataset.data, columns=california_dataset.feature_names
    )
    user_dict = {"HouseAge": [0, 20, 40, 60, np.inf]}

    data_t1 = data.copy()
    data_t2 = data.copy()

    # HouseAge is the median house age in the block group.
    data_t1["HouseAge"] = pd.cut(
        data["HouseAge"], bins=[0, 20, 40, 60, np.inf], include_lowest=True
    )
    data_t1["HouseAge"] = data_t1["HouseAge"].astype(str)
    data_t2["HouseAge"] = pd.cut(
        data["HouseAge"],
        bins=[0, 20, 40, 60, np.inf],
        labels=False,
        include_lowest=True,
    )

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
    # test case 1: when dataset contains na, transform method
    age_dict = {"Age": [0, 10, 20, 30, np.inf]}

    with pytest.raises(ValueError):
        transformer = ArbitraryDiscretiser(binning_dict=age_dict)
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_when_nan_introduced_during_transform():
    # test error when NA are introduced during the discretisation.
    rng = default_rng()

    # create dataframe with 2 variables, 1 normal and 1 skewed
    random = skewnorm.rvs(a=-50, loc=4, size=100)
    random = random - min(random)  # Shift so the minimum value is equal to zero.

    train = pd.concat(
        [
            pd.Series(rng.standard_normal(100)),
            pd.Series(random),
        ],
        axis=1,
    )

    train.columns = ["var_a", "var_b"]

    # create a dataframe with 2 variables normally distributed
    test = pd.concat(
        [
            pd.Series(rng.standard_normal(100)),
            pd.Series(rng.standard_normal(100)),
        ],
        axis=1,
    )

    test.columns = ["var_a", "var_b"]

    msg = (
        "During the discretisation, NaN values were introduced "
        "in the feature(s) var_b."
    )

    limits_dict = {"var_a": [-5, -2, 0, 2, 5], "var_b": [0, 2, 5]}

    # check for warning when errors equals 'ignore'
    with pytest.warns(UserWarning) as record:
        transformer = ArbitraryDiscretiser(binning_dict=limits_dict, errors="ignore")
        transformer.fit(train)
        transformer.transform(test)

    # check that only one warning was returned
    assert len(record) == 1
    # check that message matches
    assert record[0].message.args[0] == msg

    # check for error when errors equals 'raise'
    with pytest.raises(ValueError) as record:
        transformer = ArbitraryDiscretiser(binning_dict=limits_dict, errors="raise")
        transformer.fit(train)
        transformer.transform(test)

    # check that error message matches
    assert str(record.value) == msg


def test_error_if_not_permitted_value_is_errors():
    age_dict = {"Age": [0, 10, 20, 30, np.inf]}
    with pytest.raises(ValueError):
        ArbitraryDiscretiser(binning_dict=age_dict, errors="medialuna")


@pytest.mark.parametrize("binning_dict", ["HOLA", 1, False])
def test_error_if_binning_dict_not_dict_type(binning_dict):
    msg = (
        "binning_dict must be a dictionary with the interval limits per "
        f"variable. Got {binning_dict} instead."
    )
    with pytest.raises(ValueError) as record:
        ArbitraryDiscretiser(binning_dict=binning_dict)

    # check that error message matches
    assert str(record.value) == msg
