import re

import narwhals as nw
import pandas as pd
import pytest

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from tests.backend_helpers import frame_to_dict
from tests.estimator_checks.non_fitted_error_checks import check_raises_non_fitted_error

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)
MSG_INF = (
    "Some of the variables to transform contain inf values. Check and "
    "remove those before using this transformer."
)


class MockClass(BaseNumericalTransformer):
    def __init__(self, variables=None, return_empty=False):
        self.variables = variables
        self.return_empty = return_empty

    def fit(self, X):
        _, variables_ = self._fit_setup(X)
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def transform(self, X):
        return self._check_transform_input_and_state(X).to_native()


# fit and transform
def test_fit_setup_returns_narwhals_frame_and_numerical_variables(make_df):
    nw_X, variables_ = MockClass()._fit_setup(make_df(DATA))

    assert isinstance(nw_X, nw.DataFrame)
    assert isinstance(nw_X.to_native(), make_df)
    assert frame_to_dict(nw_X.to_native()) == DATA
    assert variables_ == ["Age", "Marks"]


def test_fit_setup_checks_user_variables(make_df):
    _, variables_ = MockClass(variables=["Marks"])._fit_setup(make_df(DATA))
    assert variables_ == ["Marks"]

    msg = (
        "Some of the variables are not numerical. Please cast them as numerical "
        "before using this transformer."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        MockClass(variables=["Name"])._fit_setup(make_df(DATA))


def test_fit_setup_when_there_are_no_numerical_variables(make_df):
    X = make_df({"Name": DATA["Name"], "City": DATA["City"]})
    msg = (
        "No numerical variables found in this dataframe. Check variable dtypes or "
        "set return_empty to True to return an empty list instead."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        MockClass()._fit_setup(X)

    msg = "No numerical variables found in this dataframe. Returning an empty list."
    with pytest.warns(UserWarning, match=re.escape(msg)):
        _, variables_ = MockClass(return_empty=True)._fit_setup(X)
    assert variables_ == []


@pytest.mark.parametrize("value, msg", [(None, MSG_NA), (float("inf"), MSG_INF)])
def test_fit_setup_raises_error_if_na_or_inf(make_df, value, msg):
    X = make_df({**DATA, "Marks": [0.9, value, 0.7, 0.6]})
    with pytest.raises(ValueError, match=re.escape(msg)):
        MockClass()._fit_setup(X)


def test_get_feature_names_in(make_df):
    transformer = MockClass().fit(make_df(DATA))
    assert transformer.feature_names_in_ == list(DATA)
    assert transformer.n_features_in_ == 4


def test_check_transform_input_and_state_returns_narwhals_frame(make_df):
    transformer = MockClass().fit(make_df(DATA))
    reordered = make_df({k: DATA[k] for k in ["Marks", "City", "Age", "Name"]})

    nw_X = transformer._check_transform_input_and_state(reordered)

    assert isinstance(nw_X, nw.DataFrame)
    assert isinstance(nw_X.to_native(), make_df)
    # the columns are returned in the order seen in fit
    assert nw_X.columns == list(DATA)
    assert frame_to_dict(nw_X.to_native()) == DATA


def test_check_transform_input_and_state_with_integer_column_names():
    # integer column names are pandas-only
    X = pd.DataFrame({0: [1.0, 2.0], 1: ["a", "b"], 2: [3, 4]})
    transformer = MockClass().fit(X)

    nw_X = transformer._check_transform_input_and_state(X[[2, 0, 1]])

    pd.testing.assert_frame_equal(nw_X.to_native(), X)


def test_check_transform_input_and_state_raises_error_if_columns_differ(make_df):
    transformer = MockClass().fit(make_df(DATA))
    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer (when using the fit() method)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer._check_transform_input_and_state(
            make_df({k: DATA[k] for k in ["Age", "Marks"]})
        )


@pytest.mark.parametrize("value, msg", [(None, MSG_NA), (float("inf"), MSG_INF)])
def test_check_transform_input_and_state_raises_error_if_na_or_inf(
    make_df, value, msg
):
    transformer = MockClass().fit(make_df(DATA))
    X = make_df({**DATA, "Marks": [0.9, value, 0.7, 0.6]})
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer._check_transform_input_and_state(X)


def test_raises_non_fitted_error():
    check_raises_non_fitted_error(MockClass())
