import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.dataframe_checks import check_X

VARTYPES_DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
    "dob": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
}
variables_str = list(VARTYPES_DATA.keys())


class MockTransformer(BaseEstimator, GetFeatureNamesOutMixin):
    def fit(self, X, y=None):
        X = check_X(X)
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X.copy()


def test_non_fitted_error():
    transformer = MockTransformer()
    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()


# ======== Tests for transformers that do not add new features to the data ========


def test_when_input_is_pandas_columns():
    df = pd.DataFrame(VARTYPES_DATA)
    transformer = MockTransformer()
    transformer.fit(df)
    assert (
        transformer.get_feature_names_out(input_features=df.columns) == variables_str
    )


def test_when_input_is_polars_columns():
    # polars' .columns is already a plain list, so this exercises the
    # `isinstance(input_features, list)` branch, not `nwd.is_pandas_index`.
    df = pl.DataFrame(VARTYPES_DATA)
    transformer = MockTransformer()
    transformer.fit(df)
    assert (
        transformer.get_feature_names_out(input_features=df.columns) == variables_str
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_with_df(make_df, input_features):
    # When the data used to train the class is a dataframe, the variable names are
    # stored in feature_names_in_. Those should be returned by get_feature_names_out()
    df = make_df(VARTYPES_DATA)
    transformer = MockTransformer()
    transformer.fit(df)
    assert (
        transformer.get_feature_names_out(input_features=input_features)
        == transformer.feature_names_in_
    )
    assert (
        transformer.get_feature_names_out(input_features=input_features)
        == variables_str
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_with_pipeline_and_df(make_df, input_features):
    df = make_df(VARTYPES_DATA)
    pipe = Pipeline([("transformer", MockTransformer())])
    pipe.fit(df)
    assert (
        pipe.get_feature_names_out(input_features=input_features)
        == pipe.named_steps["transformer"].feature_names_in_
    )
    assert pipe.get_feature_names_out(input_features=input_features) == variables_str


@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_with_pipe_and_skl_transformer_input_df(input_features):
    # SimpleImputer outputs a numpy array by default, which check_X now
    # rejects, so it must be configured to output a dataframe.
    df = pd.DataFrame(VARTYPES_DATA)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant").set_output(transform="pandas")),
            ("transformer", MockTransformer()),
        ]
    )
    pipe.fit(df)
    assert pipe.get_feature_names_out(input_features=input_features) == variables_str


def test_pipe_with_skl_transformer_that_adds_features():
    df = pd.DataFrame({"Age": VARTYPES_DATA["Age"], "Marks": VARTYPES_DATA["Marks"]})
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures().set_output(transform="pandas")),
            ("transformer", MockTransformer()),
        ]
    )
    pipe.fit(df)
    assert pipe.get_feature_names_out(input_features=None) == [
        "1",
        "Age",
        "Marks",
        "Age^2",
        "Age Marks",
        "Marks^2",
    ]

    assert pipe.get_feature_names_out(input_features=["Age", "Marks"]) == [
        "1",
        "Age",
        "Marks",
        "Age^2",
        "Age Marks",
        "Marks^2",
    ]


def test_raise_error_when_input_feature_non_permitted():
    df = pd.DataFrame(VARTYPES_DATA)
    transformer = MockTransformer()
    transformer.fit(df)

    with pytest.raises(ValueError, match="feature_names_in_"):
        transformer.get_feature_names_out(input_features=["Name"])

    with pytest.raises(ValueError, match="feature_names_in_"):
        transformer.get_feature_names_out(input_features=np.array(["Name", "Age"]))

    with pytest.raises(ValueError, match="list or an array"):
        transformer.get_feature_names_out(input_features="var1")

    with pytest.raises(ValueError, match="list or an array"):
        transformer.get_feature_names_out(input_features=True)


# ================ Tests for transformers that add features to the data =======


class MockCreator(BaseEstimator, GetFeatureNamesOutMixin):
    def __init__(self, variables, drop_original):
        self.variables = variables
        self.drop_original = drop_original

    def fit(self, X, y=None):
        X = check_X(X)
        self.variables_ = self.variables
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X.copy()

    def _get_new_features_name(self):
        return [f"{i}_plus" for i in self.variables_]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("features_in", [["Age", "Marks"], ["Name", "dob"]])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_new_feature_names_with_df(make_df, features_in, input_features):
    df = make_df(VARTYPES_DATA)
    transformer = MockCreator(variables=features_in, drop_original=False)
    transformer.fit(df)
    features_out = variables_str + [f"{i}_plus" for i in features_in]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )

    transformer = MockCreator(variables=features_in, drop_original=True)
    transformer.fit(df)
    features_out = [f for f in variables_str if f not in features_in] + [
        f"{i}_plus" for i in features_in
    ]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("features_in", [["Age", "Marks"], ["Name", "dob"]])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_new_feature_names_within_pipeline(make_df, features_in, input_features):
    df = make_df(VARTYPES_DATA)
    transformer = Pipeline(
        [
            ("transformer", MockCreator(variables=features_in, drop_original=False)),
        ]
    )
    transformer.fit(df)
    features_out = variables_str + [f"{i}_plus" for i in features_in]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )

    transformer = Pipeline(
        [
            ("transformer", MockCreator(variables=features_in, drop_original=True)),
        ]
    )
    transformer.fit(df)
    features_out = [f for f in variables_str if f not in features_in] + [
        f"{i}_plus" for i in features_in
    ]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )


@pytest.mark.parametrize("features_in", [["Age", "Marks"], ["Name", "dob"]])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_new_feature_names_pipe_with_skl_transformer_and_df(features_in, input_features):
    df = pd.DataFrame(VARTYPES_DATA)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant").set_output(transform="pandas")),
            ("transformer", MockCreator(variables=features_in, drop_original=False)),
        ]
    )
    pipe.fit(df)
    features_out = variables_str + [f"{i}_plus" for i in features_in]
    assert pipe.get_feature_names_out(input_features=input_features) == features_out

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant").set_output(transform="pandas")),
            ("transformer", MockCreator(variables=features_in, drop_original=True)),
        ]
    )
    pipe.fit(df)
    features_out = [f for f in variables_str if f not in features_in] + [
        f"{i}_plus" for i in features_in
    ]
    assert pipe.get_feature_names_out(input_features=input_features) == features_out


@pytest.mark.parametrize(
    "input_features", [None, ["Age", "Marks"], np.array(["Age", "Marks"])]
)
def test_new_feature_names_pipe_and_skl_transformer_that_adds_features(input_features):
    features_in = ["Age", "Marks"]
    df = pd.DataFrame({"Age": VARTYPES_DATA["Age"], "Marks": VARTYPES_DATA["Marks"]})

    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures().set_output(transform="pandas")),
            ("transformer", MockCreator(variables=features_in, drop_original=False)),
        ]
    )
    pipe.fit(df)

    new_features = [f"{i}_plus" for i in features_in]
    assert (
        pipe.get_feature_names_out(input_features=input_features)
        == ["1", "Age", "Marks", "Age^2", "Age Marks", "Marks^2"] + new_features
    )


# ================ Tests for transformers that remove features to the data =======


class MockSelector(BaseEstimator, GetFeatureNamesOutMixin):
    def fit(self, X, y=None):
        X = check_X(X)
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        self.features_to_drop_ = list(X.columns)[0:2]
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop_)

    def get_support(self, indices=False):
        mask = [
            True if f not in self.features_to_drop_ else False
            for f in self.feature_names_in_
        ]
        return mask if not indices else np.where(mask)[0]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_remove_features_in_df(make_df, input_features):
    df = make_df(VARTYPES_DATA)
    transformer = MockSelector()
    transformer.fit(df)
    features_out = variables_str[2:]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_remove_feature_names_within_pipeline_when_df(make_df, input_features):
    df = make_df(VARTYPES_DATA)
    transformer = Pipeline([("transformer", MockSelector())])
    transformer.fit(df)
    features_out = variables_str[2:]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )


@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_remove_feature_names_pipe_with_skl_transformer_and_df(input_features):
    df = pd.DataFrame(
        {k: v for k, v in VARTYPES_DATA.items() if k != "dob"}
    )
    variables_no_dob = [v for v in variables_str if v != "dob"]
    trimmed_input_features = (
        input_features[0:-1] if input_features is not None else None
    )

    pipe = Pipeline(
        [
            ("transformer", MockSelector()),
            ("imputer", SimpleImputer(strategy="constant").set_output(transform="pandas")),
        ]
    )
    pipe.fit(df)
    features_out = variables_no_dob[2:]
    # sklearn's Pipeline.get_feature_names_out() returns a numpy array here
    # when the feature-removing transformer isn't the last step.
    assert all(
        pipe.get_feature_names_out(input_features=trimmed_input_features)
        == features_out
    )

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant").set_output(transform="pandas")),
            ("transformer", MockSelector()),
        ]
    )
    pipe.fit(df)
    assert (
        pipe.get_feature_names_out(input_features=trimmed_input_features)
        == features_out
    )


@pytest.mark.parametrize(
    "input_features", [None, ["Age", "Marks"], np.array(["Age", "Marks"])]
)
def test_remove_feature_names_pipe_and_skl_transformer_that_adds_features(
    input_features,
):
    features_in = ["Age", "Marks"]
    df = pd.DataFrame({"Age": VARTYPES_DATA["Age"], "Marks": VARTYPES_DATA["Marks"]})

    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures().set_output(transform="pandas")),
            ("transformer", MockSelector()),
        ]
    )
    pipe.fit(df)

    assert pipe.get_feature_names_out(input_features=input_features) == [
        "Marks",
        "Age^2",
        "Age Marks",
        "Marks^2",
    ]
