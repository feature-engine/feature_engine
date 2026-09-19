import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.selection.base_selector import BaseSelector
from tests.backend_helpers import frame_to_dict

DATA = {
    "Name": ["tom", "nick", "krish", None],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, None, 0.7, 0.6],
    "dob": [datetime(2020, 2, 24, 0, minute) for minute in range(4)],
}


# init parameters
@pytest.mark.parametrize("confirm_variables", [None, "hola", [True], 1, 0.5])
def test_error_if_confirm_variables_not_bool(confirm_variables):
    msg = (
        "confirm_variables takes only values True and False. "
        f"Got {confirm_variables} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseSelector(confirm_variables=confirm_variables)


@pytest.mark.parametrize("confirm_variables", [True, False])
def test_init_param_assignment(confirm_variables):
    selector = BaseSelector(confirm_variables=confirm_variables)
    assert selector.confirm_variables is confirm_variables


# fit and transform
class MockSelector(BaseSelector):
    def __init__(self, features_to_drop=("Name", "Marks")):
        self.features_to_drop = features_to_drop
        self.confirm_variables = False

    def fit(self, X, y=None):
        self.features_to_drop_ = list(self.features_to_drop)
        self._get_feature_names_in(X)
        return self


def test_transform_drops_features(make_df):
    X = make_df(DATA)
    Xt = MockSelector().fit(X).transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "dob": DATA["dob"],
    }


def test_transform_restores_train_column_order(make_df):
    selector = MockSelector().fit(make_df(DATA))
    X = make_df({var: DATA[var] for var in ["dob", "Marks", "Age", "Name", "City"]})
    Xt = selector.transform(X)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["City", "Age", "dob"]


@pytest.mark.parametrize(
    "features_to_drop, expected",
    [
        ([], ["Name", "City", "Age", "Marks", "dob"]),
        (["dob"], ["Name", "City", "Age", "Marks"]),
        (["Age", "Name", "City", "dob"], ["Marks"]),
    ],
)
def test_transform_returns_retained_features(make_df, features_to_drop, expected):
    X = make_df(DATA)
    Xt = MockSelector(features_to_drop).fit(X).transform(X)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == expected


def test_transform_does_not_modify_input(make_df):
    X = make_df(DATA)
    MockSelector().fit(X).transform(X)
    assert frame_to_dict(X) == frame_to_dict(make_df(DATA))


def test_error_if_transform_df_has_different_number_of_columns(make_df):
    selector = MockSelector().fit(make_df(DATA))
    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer (when using the fit() method)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        selector.transform(make_df({"Age": DATA["Age"], "Marks": DATA["Marks"]}))


def test_error_if_transform_before_fit(make_df):
    msg = (
        "This MockSelector instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        MockSelector().transform(make_df(DATA))


def test_error_if_transform_input_not_dataframe(make_df):
    selector = MockSelector().fit(make_df(DATA))
    msg = (
        "X must be a dataframe from a library supported by narwhals "
        "(e.g. pandas, polars, PyArrow). Got <class 'numpy.ndarray'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        selector.transform(np.ones((4, 5)))


def test_get_feature_names_in(make_df):
    selector = MockSelector()
    selector._get_feature_names_in(make_df(DATA))
    assert selector.feature_names_in_ == ["Name", "City", "Age", "Marks", "dob"]
    assert selector.n_features_in_ == 5


def test_get_support(make_df):
    selector = MockSelector().fit(make_df(DATA))
    assert selector.get_support() == [False, True, True, False, True]
    assert list(selector.get_support(indices=True)) == [1, 2, 4]


def test_get_feature_names_out(make_df):
    selector = MockSelector().fit(make_df(DATA))
    assert selector.get_feature_names_out() == ["City", "Age", "dob"]


def test_check_variable_number():
    selector = MockSelector()
    selector.variables_ = ["Age"]
    msg = (
        "The selector needs at least 2 or more variables to select from. "
        "Got only 1 variable: ['Age']."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        selector._check_variable_number()


def test_transform_with_integer_column_names():
    X = pd.DataFrame({i: values for i, values in enumerate(DATA.values())})
    selector = MockSelector(features_to_drop=[0, 3]).fit(X)
    Xt = selector.transform(X[[4, 3, 2, 1, 0]])

    assert selector.feature_names_in_ == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(Xt, X[[1, 2, 4]])


def test_transform_keeps_pandas_index():
    X = pd.DataFrame(DATA, index=[10, 11, 12, 13])
    Xt = MockSelector().fit(X).transform(X)
    pd.testing.assert_frame_equal(Xt, X[["City", "Age", "dob"]])
