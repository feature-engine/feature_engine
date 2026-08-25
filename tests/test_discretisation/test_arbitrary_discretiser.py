import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import fetch_california_housing

from feature_engine.discretisation import ArbitraryDiscretiser


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_arbitrary_discretiser(make_df):
    california_dataset = fetch_california_housing()
    data_pd = pd.DataFrame(
        california_dataset.data, columns=california_dataset.feature_names
    )
    user_dict = {"HouseAge": [0, 20, 40, 60, np.inf]}

    # ground truth via pandas.cut - bins are user-supplied and fixed, so both
    # backends must reproduce this exact output.
    expected_codes = pd.cut(
        data_pd["HouseAge"],
        bins=[0, 20, 40, 60, np.inf],
        labels=False,
        include_lowest=True,
    ).to_numpy()
    expected_labels = (
        pd.cut(data_pd["HouseAge"], bins=[0, 20, 40, 60, np.inf], include_lowest=True)
        .astype(str)
        .to_numpy()
    )

    data = make_df(data_pd)

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
    result_codes = nw.from_native(X, eager_only=True).get_column("HouseAge").to_numpy()
    assert np.array_equal(result_codes, expected_codes)

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_object=False, return_boundaries=True
    )
    X = transformer.fit_transform(data)
    result_labels = nw.from_native(X, eager_only=True).get_column("HouseAge").to_numpy()
    assert np.array_equal(result_labels, expected_labels)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_input_df_contains_na_in_transform(make_df):
    # test case 1: when dataset contains na, transform method raises
    age_dict = {"Age": [0, 10, 20, 30, np.inf]}
    data = make_df({"Age": [20.0, 21.0, 19.0, 18.0]})
    data_na = make_df({"Age": [20.0, 21.0, None, 18.0]})

    transformer = ArbitraryDiscretiser(binning_dict=age_dict)
    transformer.fit(data)
    with pytest.raises(ValueError, match="Some of the variables in the dataset"):
        transformer.transform(data_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("return_object", [False, True])
def test_error_when_nan_introduced_during_transform(make_df, return_object):
    # test warning/error when NA are introduced during the discretisation,
    # i.e. when a value in the data to transform falls outside the bin edges
    # fitted on the training data.
    train = make_df({"var_a": [-4.0, -1.0, 1.0, 4.0], "var_b": [1.0, 2.0, 3.0, 4.0]})
    test = make_df({"var_a": [-4.0, -1.0, 1.0, 4.0], "var_b": [10.0, 20.0, 30.0, 40.0]})

    msg = (
        "During the discretisation, NaN values were introduced "
        "in the feature(s) var_b."
    )

    limits_dict = {"var_a": [-5, -2, 0, 2, 5], "var_b": [0, 2, 5]}

    # check for warning when errors equals 'ignore'
    transformer = ArbitraryDiscretiser(
        binning_dict=limits_dict, return_object=return_object, errors="ignore"
    )
    transformer.fit(train)
    with pytest.warns(UserWarning, match=re.escape(msg)):
        transformer.transform(test)

    # check for error when errors equals 'raise'
    transformer = ArbitraryDiscretiser(
        binning_dict=limits_dict, return_object=return_object, errors="raise"
    )
    transformer.fit(train)
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.transform(test)


def test_error_if_not_permitted_value_is_errors():
    age_dict = {"Age": [0, 10, 20, 30, np.inf]}
    with pytest.raises(ValueError, match="errors only takes values"):
        ArbitraryDiscretiser(binning_dict=age_dict, errors="medialuna")


@pytest.mark.parametrize("binning_dict", ["HOLA", 1, False])
def test_error_if_binning_dict_not_dict_type(binning_dict):
    msg = (
        "binning_dict must be a dictionary with the interval limits per "
        f"variable. Got {binning_dict} instead."
    )
    with pytest.raises(ValueError, match=msg):
        ArbitraryDiscretiser(binning_dict=binning_dict)
