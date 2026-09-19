import re

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.timeseries.forecasting.base_forecast_transformers import (
    BaseForecastTransformer,
)
from tests.backend_helpers import frame_to_dict, make_series

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)
MSG_INF = (
    "Some of the variables to transform contain inf values. Check and "
    "remove those before using this transformer."
)
MSG_INDEX_NA = (
    "The dataframe's index contains NaN values or missing data. Only dataframes "
    "with complete indexes are compatible with this transformer."
)
MSG_INDEX_DUPLICATED = (
    "The dataframe's index does not contain unique values. Only dataframes with "
    "unique values in the index are compatible with this transformer."
)

DATA = {
    "a": [1.0, 2.0, 3.0, 4.0],
    "b": [10, 20, 30, 40],
    "c": ["w", "x", "y", "z"],
}


class MockLag(BaseForecastTransformer):
    # adds the value of the previous row, to test the shared logic
    def __init__(
        self,
        variables=None,
        return_empty=False,
        missing_values="raise",
        drop_original=False,
        drop_na=False,
        freq=None,
        sort_index=True,
    ):
        super().__init__(
            variables, return_empty, missing_values, drop_original, drop_na
        )
        self.freq = freq
        self.sort_index = sort_index

    def _add_features(self, nw_X):
        # get_column also works with the integer column names of pandas
        return nw_X.with_columns(
            nw_X.get_column(var).shift(1).alias(name)
            for var, name in zip(self.variables_, self._get_new_features_name())
        )

    def _get_new_features_name(self):
        return [f"{var}_lag_1" for var in self.variables_]


# init parameters
@pytest.mark.parametrize("missing_values", ["other", "Raise", 1, True, None, ["raise"]])
def test_error_if_missing_values_not_permitted(missing_values):
    msg = (
        "missing_values takes only values 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseForecastTransformer(missing_values=missing_values)


@pytest.mark.parametrize("drop_original", ["True", 1, 0, None, [True]])
def test_error_if_drop_original_not_bool(drop_original):
    msg = (
        "drop_original takes only boolean values True and False. "
        f"Got {drop_original} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseForecastTransformer(drop_original=drop_original)


@pytest.mark.parametrize("drop_na", ["True", 1, 0, None, [True]])
def test_error_if_drop_na_not_bool(drop_na):
    msg = f"drop_na takes only boolean values True and False. Got {drop_na} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseForecastTransformer(drop_na=drop_na)


@pytest.mark.parametrize("return_empty", ["True", 1, 0, None, [True]])
def test_error_if_return_empty_not_bool(return_empty):
    msg = (
        "return_empty takes only boolean values True and False. "
        f"Got {return_empty} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseForecastTransformer(return_empty=return_empty)


@pytest.mark.parametrize(
    "missing_values, drop_original, drop_na",
    [("raise", False, False), ("ignore", True, False), ("raise", False, True)],
)
def test_init_param_assignment(missing_values, drop_original, drop_na):
    transformer = BaseForecastTransformer(
        missing_values=missing_values,
        drop_original=drop_original,
        drop_na=drop_na,
    )
    assert transformer.missing_values == missing_values
    assert transformer.drop_original == drop_original
    assert transformer.drop_na == drop_na


# fit and transform
def test_fit_finds_numerical_variables(make_df):
    transformer = MockLag().fit(make_df(DATA))

    assert transformer.variables_ == ["a", "b"]
    assert transformer.feature_names_in_ == ["a", "b", "c"]
    assert transformer.n_features_in_ == 3


def test_fit_checks_variables_entered_by_user(make_df):
    transformer = MockLag(variables="b").fit(make_df(DATA))
    assert transformer.variables_ == ["b"]


@pytest.mark.parametrize("y", [[1, 2, 3, 4], np.array([1, 2, 3, 4])])
def test_fit_accepts_y_as_list_and_array(make_df, y):
    transformer = MockLag().fit(make_df(DATA), y)
    assert transformer.variables_ == ["a", "b"]


def test_fit_and_transform_raise_error_if_na(make_df):
    X_na = make_df({**DATA, "a": [1.0, None, 3.0, 4.0]})
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        MockLag().fit(X_na)

    transformer = MockLag().fit(make_df(DATA))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(X_na)


def test_fit_and_transform_raise_error_if_inf(make_df):
    X_inf = make_df({**DATA, "a": [1.0, np.inf, 3.0, 4.0]})
    with pytest.raises(ValueError, match=re.escape(MSG_INF)):
        MockLag().fit(X_inf)

    transformer = MockLag().fit(make_df(DATA))
    with pytest.raises(ValueError, match=re.escape(MSG_INF)):
        transformer.transform(X_inf)


def test_transform_adds_features(make_df):
    X = make_df(DATA)
    Xt = MockLag().fit(X).transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "a_lag_1": [None, 1.0, 2.0, 3.0],
        "b_lag_1": [None, 10, 20, 30],
    }


def test_transform_keeps_na_when_ignored(make_df):
    X = make_df({**DATA, "a": [1.0, None, 3.0, 4.0]})
    Xt = MockLag(variables="a", missing_values="ignore").fit(X).transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)["a_lag_1"] == [None, 1.0, None, 3.0]


def test_transform_reorders_columns_to_match_fit(make_df):
    transformer = MockLag().fit(make_df(DATA))
    Xt = transformer.transform(make_df({k: DATA[k] for k in ["c", "b", "a"]}))

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["a", "b", "c", "a_lag_1", "b_lag_1"]


def test_drop_original(make_df):
    X = make_df(DATA)
    Xt = MockLag(drop_original=True).fit(X).transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "c": ["w", "x", "y", "z"],
        "a_lag_1": [None, 1.0, 2.0, 3.0],
        "b_lag_1": [None, 10, 20, 30],
    }


def test_drop_na(make_df):
    X = make_df({**DATA, "a": [1.0, 2.0, None, 4.0]})
    transformer = MockLag(missing_values="ignore", drop_na=True).fit(X)
    Xt = transformer.transform(X)

    assert isinstance(Xt, make_df)
    # only missing data in the new features removes a row
    assert frame_to_dict(Xt) == {
        "a": [2.0, None],
        "b": [20, 30],
        "c": ["x", "y"],
        "a_lag_1": [1.0, 2.0],
        "b_lag_1": [10, 20],
    }


def test_drop_na_without_variables_keeps_all_rows(make_df):
    X = make_df({"c": DATA["c"]})
    transformer = MockLag(return_empty=True, drop_na=True).fit(X)
    y = make_series(make_df, [1, 2, 3, 4])

    Xt, yt = transformer.transform_x_y(X, y)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(transformer.transform(X)) == {"c": DATA["c"]}
    assert frame_to_dict(Xt) == {"c": DATA["c"]}
    assert list(yt) == [1, 2, 3, 4]


def test_transform_raises_error_if_different_number_of_columns(make_df):
    transformer = MockLag().fit(make_df(DATA))
    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer (when using the fit() method)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.transform(make_df({"a": DATA["a"], "b": DATA["b"]}))


@pytest.mark.parametrize("method", ["transform", "transform_x_y"])
def test_raises_non_fitted_error(make_df, method):
    X = make_df(DATA)
    msg = (
        "This MockLag instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        if method == "transform":
            MockLag().transform(X)
        else:
            MockLag().transform_x_y(X, make_series(make_df, [1, 2, 3, 4]))


@pytest.mark.parametrize(
    "drop_na, expected", [(True, [2, 3, 4]), (False, [1, 2, 3, 4])]
)
def test_transform_x_y(make_df, drop_na, expected):
    X = make_df(DATA)
    y = make_series(make_df, [1, 2, 3, 4], name="y")
    transformer = MockLag(drop_na=drop_na).fit(X)

    Xt, yt = transformer.transform_x_y(X, y)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == frame_to_dict(transformer.transform(X))
    assert list(yt) == expected


def test_transform_x_y_with_multioutput_target(make_df):
    X = make_df(DATA)
    y = make_df({"y1": [1, 2, 3, 4], "y2": [5, 6, 7, 8]})

    Xt, yt = MockLag(drop_na=True).fit(X).transform_x_y(X, y)

    assert isinstance(yt, make_df)
    assert frame_to_dict(yt) == {"y1": [2, 3, 4], "y2": [6, 7, 8]}
    assert Xt.shape == (3, 5)


# pandas: the index orders the rows in time
DATES = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])


@pytest.fixture
def df_shuffled():
    # the rows are not in time order
    return pd.DataFrame(DATA, index=DATES).iloc[[2, 0, 3, 1]]


def test_sort_index_orders_rows_by_the_index_with_pandas(df_shuffled):
    Xt = MockLag(variables="a").fit(df_shuffled).transform(df_shuffled)

    expected = pd.DataFrame({**DATA, "a_lag_1": [np.nan, 1.0, 2.0, 3.0]}, index=DATES)
    pd.testing.assert_frame_equal(Xt, expected)
    # the input dataframe is not sorted in place
    assert list(df_shuffled.index) == list(DATES[[2, 0, 3, 1]])


def test_rows_keep_their_order_without_sort_index_with_pandas(df_shuffled):
    transformer = MockLag(variables="a", sort_index=False).fit(df_shuffled)
    Xt = transformer.transform(df_shuffled)

    expected = df_shuffled.assign(a_lag_1=[np.nan, 3.0, 1.0, 4.0])
    pd.testing.assert_frame_equal(Xt, expected)


def test_transform_x_y_aligns_y_on_the_index_with_pandas(df_shuffled):
    y = pd.Series([3, 1, 4, 2], index=df_shuffled.index, name="y")
    transformer = MockLag(variables="a", drop_na=True).fit(df_shuffled)

    Xt, yt = transformer.transform_x_y(df_shuffled, y)

    pd.testing.assert_frame_equal(Xt, transformer.transform(df_shuffled))
    pd.testing.assert_series_equal(yt, pd.Series([2, 3, 4], index=DATES[1:], name="y"))


@pytest.mark.parametrize(
    "index, msg",
    [
        (list(DATES[:3]) + [pd.NaT], MSG_INDEX_NA),
        (list(DATES[:3]) + [DATES[0]], MSG_INDEX_DUPLICATED),
    ],
)
def test_error_if_index_incomplete_or_duplicated_with_pandas(index, msg):
    X_wrong = pd.DataFrame(DATA, index=index)
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        MockLag().fit(X_wrong)

    transformer = MockLag().fit(pd.DataFrame(DATA, index=DATES))
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        transformer.transform(X_wrong)


def test_integer_column_names_with_pandas():
    X = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [4, 5, 6], "c": ["x", "y", "z"]})
    transformer = MockLag(drop_original=True, drop_na=True).fit(X)
    Xt = transformer.transform(X[["c", 1, 0]])

    expected = pd.DataFrame(
        {"c": ["y", "z"], "0_lag_1": [1.0, 2.0], "1_lag_1": [4.0, 5.0]}, index=[1, 2]
    )
    # the columns started as a mix of integers and strings
    expected.columns = expected.columns.astype(object)
    pd.testing.assert_frame_equal(Xt, expected)


# polars: rows are used in the order given
def test_sort_index_is_ignored_with_polars():
    X = pl.DataFrame({"a": [3.0, 1.0, 4.0, 2.0], "b": [30, 10, 40, 20]})
    Xt = MockLag(variables="a", sort_index=True).fit(X).transform(X)

    assert frame_to_dict(Xt)["a_lag_1"] == [None, 3.0, 1.0, 4.0]


def test_error_if_freq_with_polars():
    X = pl.DataFrame(DATA)
    msg = (
        "freq is only supported with pandas dataframes, because it uses the "
        "dataframe's DatetimeIndex. With other dataframes, leave freq=None "
        "to shift the rows by periods. Got 1D instead."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        MockLag(freq="1D").fit(X)

    # a transformer fitted on pandas can't transform polars with freq either
    transformer = MockLag(freq="1D").fit(pd.DataFrame(DATA, index=DATES))
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        transformer.transform(X)
