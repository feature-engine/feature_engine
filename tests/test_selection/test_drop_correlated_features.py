import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_engine.selection import DropCorrelatedFeatures
from tests.estimator_checks.init_params_allowed_values_checks import (
    check_error_param_confirm_variables,
    check_error_param_missing_values,
)


@pytest.fixture(scope="module")
def df_correlated_single():
    # create array with 4 correlated features and 2 independent ones
    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform array into pandas df
    colnames = ["var_" + str(i) for i in range(6)]
    X = pd.DataFrame(X, columns=colnames)

    return X


@pytest.fixture(scope="module")
def df_correlated_double():
    # create array with 8 correlated features and 4 independent ones
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform array into pandas df
    colnames = ["var_" + str(i) for i in range(12)]
    X = pd.DataFrame(X, columns=colnames)

    return X


_input_params = [
    (None, "pearson", 0.8, "ignore", False),
    ("var1", "kendall", 0.5, "raise", True),
    (["var1", "var2"], "spearman", 0.4, "raise", False),
]


@pytest.mark.parametrize(
    "_variables, _method, _threshold, _missing_values, _confirm_vars", _input_params
)
def test_input_params_assignment(
    _variables, _method, _threshold, _missing_values, _confirm_vars
):
    sel = DropCorrelatedFeatures(
        variables=_variables,
        method=_method,
        threshold=_threshold,
        missing_values=_missing_values,
        confirm_variables=_confirm_vars,
    )

    assert sel.variables == _variables
    assert sel.method == _method
    assert sel.threshold == _threshold
    assert sel.missing_values == _missing_values
    assert sel.confirm_variables == _confirm_vars


@pytest.mark.parametrize("_threshold", [3, "0.1", -0, 2, 0, 3, 1])
def test_raises_error_when_threshold_not_permitted(_threshold):
    msg = f"`threshold` must be a float between 0 and 1. Got {_threshold} instead."
    with pytest.raises(ValueError) as record:
        DropCorrelatedFeatures(threshold=_threshold)
    assert record.value.args[0] == msg


def test_error_param_missing_values():
    check_error_param_missing_values(DropCorrelatedFeatures())


def test_error_param_confirm_variables():
    check_error_param_confirm_variables(DropCorrelatedFeatures())


def test_default_params(df_correlated_single):
    transformer = DropCorrelatedFeatures(
        variables=None, method="pearson", threshold=0.8
    )
    X = transformer.fit_transform(df_correlated_single)

    # expected result
    df = df_correlated_single.drop("var_2", axis=1)

    # test fit attrs
    assert transformer.features_to_drop_ == ["var_2"]
    assert transformer.correlated_feature_sets_ == [{"var_1", "var_2"}]
    assert transformer.correlated_feature_dict_ == {"var_1": {"var_2"}}
    # test transform output
    pd.testing.assert_frame_equal(X, df)


def test_default_params_different_var_order(df_correlated_single):
    transformer = DropCorrelatedFeatures(
        variables=None, method="pearson", threshold=0.8
    )
    var_order = list(reversed(list(df_correlated_single.columns)))
    X = transformer.fit_transform(df_correlated_single[var_order])

    # expected result
    df = df_correlated_single[var_order].drop("var_2", axis=1)

    # test fit attrs
    assert transformer.features_to_drop_ == ["var_2"]
    assert transformer.correlated_feature_sets_ == [{"var_1", "var_2"}]
    assert transformer.correlated_feature_dict_ == {"var_1": {"var_2"}}
    # test transform output
    pd.testing.assert_frame_equal(X, df)


def test_lower_threshold(df_correlated_single):
    transformer = DropCorrelatedFeatures(
        variables=None, method="pearson", threshold=0.6
    )
    X = transformer.fit_transform(df_correlated_single)

    # expected result
    df = df_correlated_single.drop(["var_2", "var_4"], axis=1)

    # test fit attrs
    assert transformer.features_to_drop_ == ["var_2", "var_4"]
    assert transformer.correlated_feature_sets_ == [{"var_1", "var_2", "var_4"}]
    assert transformer.correlated_feature_dict_ == {"var_1": {"var_2", "var_4"}}
    # test transform output
    pd.testing.assert_frame_equal(X, df)


def test_more_than_1_correlated_group(df_correlated_double):
    transformer = DropCorrelatedFeatures(
        variables=None, method="pearson", threshold=0.6
    )
    X = transformer.fit_transform(df_correlated_double)

    # expected result
    df = df_correlated_double.drop(["var_6", "var_7", "var_8", "var_9"], axis=1)

    # test fit attrs
    assert transformer.features_to_drop_ == ["var_8", "var_6", "var_7", "var_9"]
    assert transformer.correlated_feature_sets_ == [
        {"var_0", "var_8"},
        {"var_4", "var_6", "var_7", "var_9"},
    ]
    assert transformer.correlated_feature_dict_ == {
        "var_0": {"var_8"},
        "var_4": {"var_6", "var_7", "var_9"},
    }
    # test transform output
    pd.testing.assert_frame_equal(X, df)


def test_callable_method(df_correlated_double, random_uniform_method):
    X = df_correlated_double

    transformer = DropCorrelatedFeatures(
        variables=None, method=random_uniform_method, threshold=0.6
    )

    Xt = transformer.fit_transform(X)

    # test no empty dataframe
    assert not Xt.empty

    # test fit attrs
    assert len(transformer.correlated_feature_sets_) > 0
    assert len(transformer.features_to_drop_) > 0
    assert len(transformer.variables_) > 0
    assert transformer.n_features_in_ == len(X.columns)


def test_raises_error_when_method_not_permitted(df_correlated_double):

    X = df_correlated_double
    method = "hola"

    transformer = DropCorrelatedFeatures(variables=None, method=method, threshold=0.8)

    with pytest.raises(ValueError) as errmsg:
        _ = transformer.fit_transform(X)

    exceptionmsg = errmsg.value.args[0]

    assert (
        exceptionmsg
        == "method must be either 'pearson', 'spearman', 'kendall', or a callable,"
        + f" '{method}' was supplied"
    )


def test_raises_missing_data_error(df_correlated_single):
    df = df_correlated_single.copy()
    df.iloc[0, 1] = np.nan
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    sel = DropCorrelatedFeatures(missing_values="raise")
    with pytest.raises(ValueError) as record:
        sel.fit(df)
    assert record.value.args[0] == msg
