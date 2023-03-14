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


@pytest.mark.parametrize("order_by", ["nulls", "uniqu", "alpha", 1, [0]])
def test_invalid_sorting_options(order_by):
    with pytest.raises(ValueError):
        DropCorrelatedFeatures(order_by=order_by)


@pytest.mark.parametrize("threshold", [-1, 0, 2, [0.5]])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        DropCorrelatedFeatures(threshold=threshold)


@pytest.mark.parametrize("method", ["hola", 1, ["pearson"]])
def test_invalid_method(method, df_correlated_single):
    with pytest.raises(ValueError):
        DropCorrelatedFeatures(method=method).fit(df_correlated_single)


def test_invalid_combination():
    with pytest.raises(ValueError):
        DropCorrelatedFeatures(order_by="nan", missing_values="raise")


def test_ordering_variables(df_correlated_single):
    # test alphabetic
    transformer = DropCorrelatedFeatures(order_by="alphabetic")
    X = transformer._sort_variables(df_correlated_single)
    assert X.columns == ["var_0", "var_1", "var_2", "var_3", "var_4", "var_5"]
    # test nan
    transformer = DropCorrelatedFeatures(order_by="nan")
    df_correlated_single.loc[0, "var_0"] = np.nan
    df_correlated_single.loc[[1, 2], "var_1"] = np.nan
    X = transformer._sort_variables(df_correlated_single)
    assert X.columns == ['var_2', 'var_3', 'var_4', 'var_5', 'var_0', 'var_1']
    # test unique
    transformer = DropCorrelatedFeatures(order_by="unique")
    df_correlated_single.loc[[1, 2, 3], "var_3"] = 1
    assert X.columns == ['var_2', 'var_4', 'var_5', 'var_0', 'var_1', 'var_3']
