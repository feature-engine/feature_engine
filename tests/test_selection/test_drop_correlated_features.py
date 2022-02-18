import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_engine.selection import DropCorrelatedFeatures


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


def test_default_params(df_correlated_single):
    transformer = DropCorrelatedFeatures(
        variables=None, method="pearson", threshold=0.8
    )
    X = transformer.fit_transform(df_correlated_single)

    # expected result
    df = df_correlated_single.drop("var_2", axis=1)

    # test init params
    assert transformer.method == "pearson"
    assert transformer.threshold == 0.8

    # test fit attrs
    assert transformer.features_to_drop_ == {"var_2"}
    assert transformer.correlated_feature_sets_ == [{"var_1", "var_2"}]
    # test transform output
    pd.testing.assert_frame_equal(X, df)


def test_lower_threshold(df_correlated_single):
    transformer = DropCorrelatedFeatures(
        variables=None, method="pearson", threshold=0.6
    )
    X = transformer.fit_transform(df_correlated_single)

    # expected result
    df = df_correlated_single.drop(["var_2", "var_4"], axis=1)

    # test init params
    assert transformer.method == "pearson"
    assert transformer.threshold == 0.6

    # test fit attrs
    assert transformer.features_to_drop_ == {"var_2", "var_4"}
    assert transformer.correlated_feature_sets_ == [{"var_1", "var_2", "var_4"}]
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
    assert transformer.features_to_drop_ == {"var_6", "var_7", "var_8", "var_9"}
    assert transformer.correlated_feature_sets_ == [
        {"var_0", "var_8"},
        {"var_4", "var_6", "var_7", "var_9"},
    ]
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


def test_error_method_supplied(df_correlated_double):

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
