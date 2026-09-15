import re

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import GeometricWidthDiscretiser
from tests.backend_helpers import frame_to_dict

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


# test init params
@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_object_not_bool(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(return_object=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_boundaries_not_bool(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(return_boundaries=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 0, -1])
def test_raises_error_when_precision_not_int(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(precision=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}])
def test_raises_error_when_bins_not_int(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(bins=param)


@pytest.mark.parametrize("params", [(False, 1), (True, 10)])
def test_correct_param_assignment_at_init(params):
    param1, param2 = params
    t = GeometricWidthDiscretiser(
        return_object=param1, return_boundaries=param1, precision=param2, bins=param2
    )
    assert t.return_object is param1
    assert t.return_boundaries is param1
    assert t.precision == param2
    assert t.bins == param2


def test_fit_and_transform_methods(make_df, data_normal_dist):
    transformer = GeometricWidthDiscretiser(
        bins=10, variables=None, return_object=False
    )
    X = transformer.fit_transform(make_df(data_normal_dist))

    # manual calculation
    arr = np.array(data_normal_dist["var"])
    min_, max_ = arr.min(), arr.max()
    increment = np.power(max_ - min_, 1.0 / 10)
    bins = np.r_[-np.inf, min_ + np.power(increment, np.arange(1, 10)), np.inf]
    bins = np.sort(bins)

    # fit params
    assert (transformer.binner_dict_["var"] == bins).all()

    # transform params - ground truth from pandas.cut on the same bins; values
    # must match regardless of which backend the input dataframe uses.
    expected = pd.cut(pd.Series(arr), bins=bins, precision=7).cat.codes.tolist()
    assert isinstance(X, make_df)
    assert frame_to_dict(X)["var"] == expected


def test_automatically_find_variables_and_return_as_object(make_df, data_normal_dist):
    transformer = GeometricWidthDiscretiser(bins=10, variables=None, return_object=True)
    X = transformer.fit_transform(make_df(data_normal_dist))
    assert isinstance(X, make_df)
    assert nw.from_native(X, eager_only=True).schema["var"] == nw.Object


def test_error_if_input_df_contains_na_in_fit(make_df):
    df_na = make_df({"Age": [20.0, 21.0, None, 23.0]})
    transformer = GeometricWidthDiscretiser()
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(df_na)


def test_error_if_input_df_contains_na_in_transform(make_df):
    df = make_df({"Age": [20.0, 21.0, 19.0, 23.0]})
    df_na = make_df({"Age": [20.0, 21.0, None, 23.0]})

    transformer = GeometricWidthDiscretiser()
    transformer.fit(df)
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(df_na)


def test_non_fitted_error(make_df):
    df = make_df({"Age": [20.0, 21.0, 19.0, 23.0]})
    transformer = GeometricWidthDiscretiser()
    with pytest.raises(NotFittedError):
        transformer.transform(df)
