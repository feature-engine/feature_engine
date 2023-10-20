import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from feature_engine.selection import SmartCorrelatedSelection


@pytest.fixture(scope="module")
def df_single():
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

    # trasform array into pandas df
    colnames = ["var_" + str(i) for i in range(6)]
    X = pd.DataFrame(X, columns=colnames)

    return X, y


def test_model_performance_single_corr_group(df_single):
    X, y = df_single

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
        cv=3,
    )

    Xt = transformer.fit_transform(X, y)

    # expected result
    df = X[["var_0", "var_2", "var_3", "var_4", "var_5"]].copy()

    # test init params
    assert transformer.method == "pearson"
    assert transformer.threshold == 0.8
    assert transformer.missing_values == "raise"
    assert transformer.selection_method == "model_performance"
    assert transformer.scoring == "roc_auc"
    assert transformer.cv == 3

    # test fit attrs
    assert transformer.correlated_feature_sets_ == [{"var_1", "var_2"}]
    assert transformer.features_to_drop_ == ["var_1"]
    # test transform output
    pd.testing.assert_frame_equal(Xt, df)


def test_model_performance_2_correlated_groups(df_test):
    X, y = df_test

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
        cv=3,
    )

    Xt = transformer.fit_transform(X, y)

    # expected result
    df = X[
        ["var_0", "var_1", "var_2", "var_3", "var_5", "var_7", "var_10", "var_11"]
    ].copy()

    # test fit attrs
    assert transformer.correlated_feature_sets_ == [
        {"var_0", "var_8"},
        {"var_4", "var_6", "var_7", "var_9"},
    ]
    assert transformer.features_to_drop_ == [
        "var_4",
        "var_6",
        "var_8",
        "var_9",
    ]
    # test transform output
    pd.testing.assert_frame_equal(Xt, df)


def test_error_if_select_model_performance_and_y_is_none(df_single):
    X, y = df_single

    transformer = SmartCorrelatedSelection(
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
    )

    with pytest.raises(ValueError):
        transformer.fit(X)


def test_variance_2_correlated_groups(df_test):
    X, y = df_test

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="variance",
        estimator=None,
    )

    Xt = transformer.fit_transform(X, y)

    # expected result
    df = X[
        ["var_1", "var_2", "var_3", "var_5", "var_7", "var_8", "var_10", "var_11"]
    ].copy()

    assert transformer.features_to_drop_ == [
        "var_0",
        "var_4",
        "var_6",
        "var_9",
    ]
    # test transform output
    pd.testing.assert_frame_equal(Xt, df)


def test_cardinality_2_correlated_groups(df_test):
    X, y = df_test
    X[["var_0", "var_6", "var_7", "var_9"]] = X[
        ["var_0", "var_6", "var_7", "var_9"]
    ].astype(int)

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="cardinality",
        estimator=None,
    )

    Xt = transformer.fit_transform(X, y)

    # expected result
    df = X[
        ["var_1", "var_2", "var_3", "var_4", "var_5", "var_8", "var_10", "var_11"]
    ].copy()

    assert transformer.features_to_drop_ == [
        "var_0",
        "var_6",
        "var_7",
        "var_9",
    ]
    # test transform output
    pd.testing.assert_frame_equal(Xt, df)


def test_automatic_variable_selection(df_test):
    X, y = df_test

    X[["var_0", "var_6", "var_7", "var_9"]] = X[
        ["var_0", "var_6", "var_7", "var_9"]
    ].astype(int)

    # add 2 additional categorical variables, these should not be evaluated by
    # the selector
    X["cat_1"] = "cat1"
    X["cat_2"] = "cat2"

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="cardinality",
        estimator=None,
    )

    Xt = transformer.fit_transform(X, y)

    # expected result
    df = X[
        [
            "var_1",
            "var_2",
            "var_3",
            "var_4",
            "var_5",
            "var_8",
            "var_10",
            "var_11",
            "cat_1",
            "cat_2",
        ]
    ].copy()

    assert transformer.features_to_drop_ == [
        "var_0",
        "var_6",
        "var_7",
        "var_9",
    ]
    # test transform output
    pd.testing.assert_frame_equal(Xt, df)


def test_callable_method(df_test, random_uniform_method):
    X, _ = df_test

    transformer = SmartCorrelatedSelection(
        method=random_uniform_method,
    )

    Xt = transformer.fit_transform(X)

    # test no empty dataframe
    assert not Xt.empty

    # test fit attrs
    assert len(transformer.correlated_feature_sets_) > 0
    assert len(transformer.features_to_drop_) > 0
    assert len(transformer.variables_) > 0
    assert transformer.n_features_in_ == len(X.columns)


def test_raises_param_errors():
    with pytest.raises(ValueError):
        SmartCorrelatedSelection(threshold=None)

    with pytest.raises(ValueError):
        SmartCorrelatedSelection(missing_values=None)

    with pytest.raises(ValueError):
        SmartCorrelatedSelection(selection_method="random")

    with pytest.raises(ValueError):
        SmartCorrelatedSelection(
            selection_method="missing_values", missing_values="raise"
        )


def test_error_method_supplied(df_test):
    X, _ = df_test
    method = "hola"

    transformer = SmartCorrelatedSelection(method=method)

    with pytest.raises(ValueError) as errmsg:
        _ = transformer.fit_transform(X)

    exceptionmsg = errmsg.value.args[0]

    assert (
            exceptionmsg
            == "method must be either 'pearson', 'spearman', 'kendall', or a callable,"
            + f" '{method}' was supplied"
    )


def test_consistent_features_to_drop_by_selection_method_missing_values():
    import subprocess
    script = """
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.selection import SmartCorrelatedSelection
import sys
# load data
data = sns.load_dataset("titanic")
data.drop(["who", "adult_male", "alive"], axis=1, inplace=True)
# preprocessing
one_hot_encode_low_cardinal=["sex", "embarked", "class", "deck", "embark_town", "alone"]
encoding_transformer = Pipeline(steps=[
    ("step_1", ArbitraryNumberImputer(variables="age")),
    ("step_2", CategoricalImputer(variables=["deck", "embark_town", "embarked"])),
    ("step_3", OneHotEncoder(variables=one_hot_encode_low_cardinal,
                             drop_last=True,
                             ignore_format=True))
])
encoded_data = encoding_transformer.fit_transform(data)
X_train = encoded_data.drop("survived", axis=1)
y_train = encoded_data["survived"]
# smartselection
tr = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.7,
    missing_values="ignore",
    selection_method="missing_values",
    estimator=None,
)
Xt = tr.fit(X_train, y_train)
sys.stdout.write(str(tr.features_to_drop_))
    """
    features_to_drop_0 = subprocess.check_output(["python", "-c", script])
    for _ in range(25):
        features_to_drop = subprocess.check_output(["python", "-c", script])
        # print(features_to_drop_0)
        assert eval(features_to_drop.decode()) == eval(features_to_drop_0.decode())


def test_consistent_features_to_drop_by_selection_method_cardinality():
    import subprocess
    script = """
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.selection import SmartCorrelatedSelection
import sys
# load data
data = sns.load_dataset("titanic")
data.drop(["who", "adult_male", "alive"], axis=1, inplace=True)
# preprocessing
one_hot_encode_low_cardinal=["sex", "embarked", "class", "deck", "embark_town", "alone"]
encoding_transformer = Pipeline(steps=[
    ("step_1", ArbitraryNumberImputer(variables="age")),
    ("step_2", CategoricalImputer(variables=["deck", "embark_town", "embarked"])),
    ("step_3", OneHotEncoder(variables=one_hot_encode_low_cardinal,
                             drop_last=True,
                             ignore_format=True))
])
encoded_data = encoding_transformer.fit_transform(data)
X_train = encoded_data.drop("survived", axis=1)
y_train = encoded_data["survived"]
# smartselection
tr = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.7,
    missing_values="raise",
    selection_method="cardinality",
    estimator=None,
)
Xt = tr.fit(X_train, y_train)
sys.stdout.write(str(tr.features_to_drop_))
    """
    features_to_drop_0 = subprocess.check_output(["python", "-c", script])
    for _ in range(25):
        features_to_drop = subprocess.check_output(["python", "-c", script])
        # print(features_to_drop_0)
        assert eval(features_to_drop.decode()) == eval(features_to_drop_0.decode())


def test_consistent_features_to_drop_by_selection_method_variance():
    import subprocess
    script = """
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.selection import SmartCorrelatedSelection
import sys
# load data
data = sns.load_dataset("titanic")
data.drop(["who", "adult_male", "alive"], axis=1, inplace=True)
# preprocessing
one_hot_encode_low_cardinal=["sex", "embarked", "class", "deck", "embark_town", "alone"]
encoding_transformer = Pipeline(steps=[
    ("step_1", ArbitraryNumberImputer(variables="age")),
    ("step_2", CategoricalImputer(variables=["deck", "embark_town", "embarked"])),
    ("step_3", OneHotEncoder(variables=one_hot_encode_low_cardinal,
                             drop_last=True,
                             ignore_format=True))
])
encoded_data = encoding_transformer.fit_transform(data)
X_train = encoded_data.drop("survived", axis=1)
y_train = encoded_data["survived"]
# smartselection
tr = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.7,
    missing_values="raise",
    selection_method="variance",
    estimator=None,
)
Xt = tr.fit(X_train, y_train)
sys.stdout.write(str(tr.features_to_drop_))
    """
    features_to_drop_0 = subprocess.check_output(["python", "-c", script])
    for _ in range(25):
        features_to_drop = subprocess.check_output(["python", "-c", script])
        # print(features_to_drop_0)
        assert eval(features_to_drop.decode()) == eval(features_to_drop_0.decode())


def test_consistent_features_to_drop_by_selection_method_model_performance():
    import subprocess
    script = """
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.selection import SmartCorrelatedSelection
from xgboost import XGBClassifier
import sys
# load data
data = sns.load_dataset("titanic")
data.drop(["who", "adult_male", "alive"], axis=1, inplace=True)
# preprocessing
one_hot_encode_low_cardinal=["sex", "embarked", "class", "deck", "embark_town", "alone"]
encoding_transformer = Pipeline(steps=[
    ("step_1", ArbitraryNumberImputer(variables="age")),
    ("step_2", CategoricalImputer(variables=["deck", "embark_town", "embarked"])),
    ("step_3", OneHotEncoder(variables=one_hot_encode_low_cardinal,
                             drop_last=True,
                             ignore_format=True))
])
encoded_data = encoding_transformer.fit_transform(data)
X_train = encoded_data.drop("survived", axis=1)
X_train = X_train.fillna(-999999999)
y_train = encoded_data["survived"]
# model init
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss'
)
# smartselection
tr = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.7,
    missing_values="raise",
    selection_method="model_performance",
    estimator=xgb_model,
)
Xt = tr.fit(X_train, y_train)
sys.stdout.write(str(tr.features_to_drop_))
    """
    features_to_drop_0 = subprocess.check_output(["python", "-c", script])
    for _ in range(25):
        features_to_drop = subprocess.check_output(["python", "-c", script])
        # print(features_to_drop)
        assert eval(features_to_drop.decode()) == eval(features_to_drop_0.decode())
