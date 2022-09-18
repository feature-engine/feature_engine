import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.dataframe_checks import check_X


class MockTransformer(GetFeatureNamesOutMixin):
    def fit(self, X, y=None):
        X = check_X(X)
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X.copy()


variables_str = ["Name", "City", "Age", "Marks", "dob"]
variables_arr = ["x0", "x1", "x2", "x3", "x4"]
variables_user = ["Dog", "Cat", "Bird", "Frog", "Duck"]

# ======== Tests for transformers that do not add new features to the data ========


@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_with_df(df_vartypes, input_features):
    # When the data used to train the class is a dataframe, the variable names are
    # stored in feature_names_in_. Those should be returned by get_feature_names_out()
    transformer = MockTransformer()
    transformer.fit(df_vartypes)
    assert (
        transformer.get_feature_names_out(input_features=input_features)
        == transformer.feature_names_in_
    )
    assert (
        transformer.get_feature_names_out(input_features=input_features)
        == variables_str
    )


@pytest.mark.parametrize(
    "input_features",
    [
        None,
        variables_arr,
        np.array(variables_arr),
        variables_str,
        np.array(variables_str),
        variables_user,
    ],
)
def test_with_array(df_vartypes, input_features):
    # When the data used to train the class is a numpy array, the names stored in
    # feature_names_in_ are x0, x1, etc. Those should be returned by
    # get_feature_names_out() when input_features is None. Alternatively, it returns
    # a list of the variables entered by the user.
    transformer = MockTransformer()
    transformer.fit(df_vartypes.to_numpy())

    if input_features is None:
        assert (
            transformer.get_feature_names_out(input_features=input_features)
            == variables_arr
        )
    else:
        assert transformer.get_feature_names_out(input_features=input_features) == list(
            input_features
        )


@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_with_pipeline_and_df(df_vartypes, input_features):
    pipe = Pipeline([("transformer", MockTransformer())])
    pipe.fit(df_vartypes)
    assert (
        pipe.get_feature_names_out(input_features=input_features)
        == pipe.named_steps["transformer"].feature_names_in_
    )
    assert pipe.get_feature_names_out(input_features=input_features) == variables_str


@pytest.mark.parametrize(
    "input_features",
    [
        None,
        variables_arr,
        np.array(variables_arr),
        variables_str,
        np.array(variables_str),
        variables_user,
    ],
)
def test_with_pipeline_and_array(df_vartypes, input_features):
    pipe = Pipeline([("transformer", MockTransformer())])
    pipe.fit(df_vartypes.to_numpy())

    if input_features is None:
        assert (
            pipe.get_feature_names_out(input_features=input_features) == variables_arr
        )
    else:
        assert pipe.get_feature_names_out(input_features=input_features) == list(
            input_features
        )


@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_with_pipe_and_skl_transformer_input_df(df_vartypes, input_features):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant")),
            ("transformer", MockTransformer()),
        ]
    )
    pipe.fit(df_vartypes)
    assert pipe.get_feature_names_out(input_features=input_features) == variables_str


@pytest.mark.parametrize(
    "input_features",
    [
        None,
        variables_arr,
        np.array(variables_arr),
        variables_str,
        np.array(variables_str),
        variables_user,
    ],
)
def test_with_pipe_and_skl_transformer_input_array(df_vartypes, input_features):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant")),
            ("transformer", MockTransformer()),
        ]
    )
    pipe.fit(df_vartypes.to_numpy())

    if input_features is None:
        assert (
            pipe.get_feature_names_out(input_features=input_features) == variables_arr
        )
    else:
        assert pipe.get_feature_names_out(input_features=input_features) == list(
            input_features
        )


def test_pipe_with_skl_transformer_that_adds_features(df_vartypes):
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures()),
            ("transformer", MockTransformer()),
        ]
    )

    # when input is array
    pipe.fit(df_vartypes[["Age", "Marks"]].to_numpy())
    assert pipe.get_feature_names_out(input_features=None) == [
        "1",
        "x0",
        "x1",
        "x0^2",
        "x0 x1",
        "x1^2",
    ]

    assert pipe.get_feature_names_out(input_features=["Age", "Marks"]) == [
        "1",
        "Age",
        "Marks",
        "Age^2",
        "Age Marks",
        "Marks^2",
    ]
    assert pipe.get_feature_names_out(input_features=["Dog", "Cat"]) == [
        "1",
        "Dog",
        "Cat",
        "Dog^2",
        "Dog Cat",
        "Cat^2",
    ]

    # when input is df
    pipe.fit(df_vartypes[["Age", "Marks"]])
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


def test_raise_error_when_input_feature_non_permitted(df_vartypes):
    transformer = MockTransformer()

    # when input is dataframe
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features=["Name"])
    assert "feature_names_in_" in str(record)

    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features=np.array(["Name", "Age"]))
    assert "feature_names_in_" in str(record)

    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features="var1")
    assert "list or an array" in str(record)

    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features=True)
    assert "list or an array" in str(record)

    # when input is array
    transformer.fit(df_vartypes.to_numpy())
    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features=["Name", "Age"])
    assert "number of input_features does not match" in str(record)


class MockCreator(GetFeatureNamesOutMixin):
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


# ================ Tests for transformers that add features to the data =======


@pytest.mark.parametrize("features_in", [["Age", "Marks"], ["Name", "dob"]])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_new_feature_names_with_df(df_vartypes, features_in, input_features):
    transformer = MockCreator(variables=features_in, drop_original=False)
    transformer.fit(df_vartypes)
    features_out = list(df_vartypes.columns) + [f"{i}_plus" for i in features_in]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )

    transformer = MockCreator(variables=features_in, drop_original=True)
    transformer.fit(df_vartypes)
    features_out = [f for f in df_vartypes.columns if f not in features_in] + [
        f"{i}_plus" for i in features_in
    ]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )


@pytest.mark.parametrize("features_in", [["Age", "Marks"], ["Name", "dob"]])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_new_feature_names_within_pipeline(df_vartypes, features_in, input_features):
    transformer = Pipeline(
        [
            ("transformer", MockCreator(variables=features_in, drop_original=False)),
        ]
    )
    transformer.fit(df_vartypes)
    features_out = list(df_vartypes.columns) + [f"{i}_plus" for i in features_in]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )

    transformer = Pipeline(
        [
            ("transformer", MockCreator(variables=features_in, drop_original=True)),
        ]
    )
    transformer.fit(df_vartypes)
    features_out = [f for f in df_vartypes.columns if f not in features_in] + [
        f"{i}_plus" for i in features_in
    ]
    assert (
        transformer.get_feature_names_out(input_features=input_features) == features_out
    )


@pytest.mark.parametrize("features_in", [["Age", "Marks"], ["Name", "dob"]])
@pytest.mark.parametrize(
    "input_features", [None, variables_str, np.array(variables_str)]
)
def test_new_feature_names_pipe_with_skl_transformer_and_df(
    df_vartypes, features_in, input_features
):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant")),
            ("transformer", MockCreator(variables=features_in, drop_original=False)),
        ]
    )
    pipe.fit(df_vartypes)
    features_out = list(df_vartypes.columns) + [f"{i}_plus" for i in features_in]
    assert pipe.get_feature_names_out(input_features=input_features) == features_out
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant")),
            ("transformer", MockCreator(variables=features_in, drop_original=True)),
        ]
    )
    pipe.fit(df_vartypes)
    features_out = [f for f in df_vartypes.columns if f not in features_in] + [
        f"{i}_plus" for i in features_in
    ]
    assert pipe.get_feature_names_out(input_features=input_features) == features_out


@pytest.mark.parametrize(
    "input_features", [None, ["Age", "Marks"], np.array(["Age", "Marks"])]
)
def test_new_feature_names_pipe_and_skl_transformer_that_adds_features(
    df_vartypes, input_features
):
    features_in = ["Age", "Marks"]
    df = df_vartypes[features_in].copy()

    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures()),
            ("transformer", MockCreator(variables=features_in, drop_original=False)),
        ]
    )
    pipe.fit(df)

    new_features = [f"{i}_plus" for i in features_in]
    assert (
        pipe.get_feature_names_out(input_features=input_features)
        == ["1", "Age", "Marks", "Age^2", "Age Marks", "Marks^2"] + new_features
    )
