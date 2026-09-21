import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_engine.selection import DropCorrelatedFeatures
from tests.backend_helpers import frame_to_dict

# with pearson, only a and c are correlated above 0.8. b is a monotonic, non-linear
# function of a, so the rank methods also find it; kendall doesn't find c.
DATA = {
    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "b": [2.7, 7.4, 20.1, 54.6, 148.4, 403.4, 1096.6, 2981.0, 8103.1, 22026.5],
    "c": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0],
    "e": [5.0, 1.0, 9.0, 2.0, 8.0, 3.0, 10.0, 4.0, 7.0, 6.0],
}


def pearson(x, y):
    return np.corrcoef(x, y)[0, 1]


@pytest.fixture(scope="module")
def data_correlated_single():
    """6 variables: var_1 is highly correlated with var_2 and less with var_4."""
    X, _ = make_classification(
        n_samples=1000,
        n_features=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )
    return {f"var_{i}": X[:, i].tolist() for i in range(6)}


@pytest.fixture(scope="module")
def data_correlated_double(data_classification):
    return {k: v for k, v in data_classification.items() if k != "target"}


# init parameters
@pytest.mark.parametrize("threshold", [3, "0.1", 0, 1, 2, -0.1, 1.5, None, [0.5], True])
def test_error_if_threshold_not_allowed(threshold):
    msg = f"`threshold` must be a float between 0 and 1. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropCorrelatedFeatures(threshold=threshold)


@pytest.mark.parametrize("missing_values", [2, "hola", False, None, ["raise"]])
def test_error_if_missing_values_not_allowed(missing_values):
    msg = (
        "`missing_values` takes only values 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropCorrelatedFeatures(missing_values=missing_values)


@pytest.mark.parametrize("confirm_variables", [2, "hola", [True], None])
def test_error_if_confirm_variables_not_bool(confirm_variables):
    msg = (
        "confirm_variables takes only values True and False. "
        f"Got {confirm_variables} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropCorrelatedFeatures(confirm_variables=confirm_variables)


@pytest.mark.parametrize(
    "method, threshold, missing_values, confirm_variables",
    [
        ("pearson", 0.8, "ignore", False),
        ("kendall", 0.5, "raise", True),
        ("spearman", 0.0, "raise", False),
        (pearson, 1.0, "ignore", True),
    ],
)
def test_init_param_assignment(method, threshold, missing_values, confirm_variables):
    sel = DropCorrelatedFeatures(
        method=method,
        threshold=threshold,
        missing_values=missing_values,
        confirm_variables=confirm_variables,
    )
    assert sel.method is method
    assert sel.threshold == threshold
    assert sel.missing_values == missing_values
    assert sel.confirm_variables is confirm_variables


# fit and transform
def test_default_params(make_df, data_correlated_single):
    X = make_df(data_correlated_single)
    sel = DropCorrelatedFeatures()
    Xt = sel.fit_transform(X)

    assert sel.variables_ == ["var_0", "var_1", "var_2", "var_3", "var_4", "var_5"]
    assert sel.features_to_drop_ == ["var_2"]
    assert sel.correlated_feature_sets_ == [{"var_1", "var_2"}]
    assert sel.correlated_feature_dict_ == {"var_1": {"var_2"}}
    assert sel.feature_names_in_ == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
    ]
    assert sel.n_features_in_ == 6
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        var: data_correlated_single[var]
        for var in ["var_0", "var_1", "var_3", "var_4", "var_5"]
    }


def test_result_does_not_depend_on_column_order(make_df, data_correlated_single):
    order = ["var_5", "var_4", "var_3", "var_2", "var_1", "var_0"]
    X = make_df({var: data_correlated_single[var] for var in order})
    sel = DropCorrelatedFeatures()
    Xt = sel.fit_transform(X)

    assert sel.features_to_drop_ == ["var_2"]
    assert sel.correlated_feature_sets_ == [{"var_1", "var_2"}]
    assert sel.correlated_feature_dict_ == {"var_1": {"var_2"}}
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_5", "var_4", "var_3", "var_1", "var_0"]


def test_lower_threshold(make_df, data_correlated_single):
    X = make_df(data_correlated_single)
    sel = DropCorrelatedFeatures(threshold=0.6)
    Xt = sel.fit_transform(X)

    assert sel.features_to_drop_ == ["var_2", "var_4"]
    assert sel.correlated_feature_sets_ == [{"var_1", "var_2", "var_4"}]
    assert sel.correlated_feature_dict_ == {"var_1": {"var_2", "var_4"}}
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_0", "var_1", "var_3", "var_5"]


def test_more_than_one_correlated_group(make_df, data_correlated_double):
    X = make_df(data_correlated_double)
    sel = DropCorrelatedFeatures(threshold=0.6)
    Xt = sel.fit_transform(X)

    assert sel.features_to_drop_ == ["var_8", "var_6", "var_7", "var_9"]
    assert sel.correlated_feature_sets_ == [
        {"var_0", "var_8"},
        {"var_4", "var_6", "var_7", "var_9"},
    ]
    assert sel.correlated_feature_dict_ == {
        "var_0": {"var_8"},
        "var_4": {"var_6", "var_7", "var_9"},
    }
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        var: data_correlated_double[var]
        for var in ["var_0", "var_1", "var_2", "var_3", "var_4", "var_5"]
        + ["var_10", "var_11"]
    }


@pytest.mark.parametrize(
    "method, features_to_drop, correlated_sets, correlated_dict, retained",
    [
        ("pearson", ["c"], [{"a", "c"}], {"a": {"c"}}, ["a", "b", "e"]),
        ("spearman", ["b", "c"], [{"a", "b", "c"}], {"a": {"b", "c"}}, ["a", "e"]),
        ("kendall", ["b"], [{"a", "b"}], {"a": {"b"}}, ["a", "c", "e"]),
        (pearson, ["c"], [{"a", "c"}], {"a": {"c"}}, ["a", "b", "e"]),
    ],
)
def test_correlation_methods(
    make_df, method, features_to_drop, correlated_sets, correlated_dict, retained
):
    sel = DropCorrelatedFeatures(method=method)
    Xt = sel.fit_transform(make_df(DATA))

    assert sel.features_to_drop_ == features_to_drop
    assert sel.correlated_feature_sets_ == correlated_sets
    assert sel.correlated_feature_dict_ == correlated_dict
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {var: DATA[var] for var in retained}


def test_negative_correlation_is_also_dropped(make_df):
    X = make_df(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [-2.0, -4.0, -6.0, -8.0, -10.0, -12.5],
            "z": [3.0, 1.0, 2.0, 6.0, 4.0, 5.0],
        }
    )
    sel = DropCorrelatedFeatures().fit(X)

    assert sel.features_to_drop_ == ["y"]
    assert sel.correlated_feature_dict_ == {"x": {"y"}}


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall", pearson])
def test_missing_values_are_ignored_pairwise(make_df, method):
    # y differs from 2 * x only in the row where x is missing.
    data = {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, None],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0, 1000.0],
        "w": [None, 3.0, 1.0, 2.0, 5.0, 4.0],
    }
    sel = DropCorrelatedFeatures(method=method, missing_values="ignore")
    Xt = sel.fit_transform(make_df(data))

    assert sel.features_to_drop_ == ["y"]
    assert sel.correlated_feature_sets_ == [{"x", "y"}]
    assert sel.correlated_feature_dict_ == {"x": {"y"}}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"x": data["x"], "w": data["w"]}


def test_non_numerical_variables_are_ignored_and_kept(make_df):
    data = {
        **DATA,
        "cat": ["x", "y"] * 5,
        "dob": [datetime(2020, 2, 24, 0, minute) for minute in range(10)],
    }
    sel = DropCorrelatedFeatures()
    Xt = sel.fit_transform(make_df(data))

    assert sel.variables_ == ["a", "b", "c", "e"]
    assert sel.features_to_drop_ == ["c"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        var: data[var] for var in ["a", "b", "e", "cat", "dob"]
    }


def test_variables_subset(make_df):
    sel = DropCorrelatedFeatures(variables=["b", "c", "e"])
    Xt = sel.fit_transform(make_df(DATA))

    assert sel.variables_ == ["b", "c", "e"]
    assert sel.features_to_drop_ == []
    assert sel.correlated_feature_sets_ == []
    assert sel.correlated_feature_dict_ == {}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == DATA


def test_confirm_variables(make_df):
    sel = DropCorrelatedFeatures(variables=["a", "c", "hola"], confirm_variables=True)
    Xt = sel.fit_transform(make_df(DATA))

    assert sel.variables_ == ["a", "c"]
    assert sel.features_to_drop_ == ["c"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["a", "b", "e"]


def test_error_if_missing_values_and_raise(make_df):
    X = make_df({**DATA, "a": [None] + DATA["a"][1:]})
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropCorrelatedFeatures(missing_values="raise").fit(X)


def test_error_if_inf_and_raise(make_df):
    X = make_df({**DATA, "a": [float("inf")] + DATA["a"][1:]})
    msg = (
        "Some of the variables to transform contain inf values. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropCorrelatedFeatures(missing_values="raise").fit(X)


def test_error_if_less_than_two_numerical_variables(make_df):
    X = make_df({"a": DATA["a"], "cat": ["x", "y"] * 5})
    msg = (
        "The selector needs at least 2 or more variables to select from. "
        "Got only 1 variable: ['a']."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropCorrelatedFeatures().fit(X)


def test_error_if_variables_not_numerical(make_df):
    X = make_df({**DATA, "cat": ["x", "y"] * 5})
    msg = (
        "Some of the variables are not numerical. Please cast them as numerical "
        "before using this transformer."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        DropCorrelatedFeatures(variables=["a", "cat"]).fit(X)


def test_error_if_fit_input_not_dataframe():
    msg = (
        "X must be a dataframe from a library supported by narwhals "
        "(e.g. pandas, polars, PyArrow). Got <class 'numpy.ndarray'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        DropCorrelatedFeatures().fit(np.ones((4, 3)))


def test_error_if_method_not_allowed_pandas():
    # the message comes from pandas.DataFrame.corr().
    msg = (
        "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        "'hola' was supplied"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropCorrelatedFeatures(method="hola").fit(pd.DataFrame(DATA))


def test_integer_column_names_pandas():
    X = pd.DataFrame({i: values for i, values in enumerate(DATA.values())})
    sel = DropCorrelatedFeatures()
    Xt = sel.fit_transform(X)

    assert sel.features_to_drop_ == [2]
    assert sel.correlated_feature_dict_ == {0: {2}}
    pd.testing.assert_frame_equal(Xt, X[[0, 1, 3]])
