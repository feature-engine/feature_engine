import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.dataframe_checks import check_X

class MockClass(GetFeatureNamesOutMixin):
    def fit(self, X, y=None):
        X = check_X(X)
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X.copy()


variables_str = ["Name", "City", "Age", "Marks", "dob"]
variables_arr = ["x0", "x1", "x2", "x3", "x4"]

@pytest.mark.parametrize("input_features", [None, variables_str, np.array(variables_str)])
def test_with_df(df_vartypes, input_features):
    # when the data used to train the class is a dataframe, the variable names are
    # stored in feature_names_in_. Those should be returned by get_feature_names_out()
    transformer = MockClass()
    transformer.fit(df_vartypes)
    assert (
        transformer.get_feature_names_out(input_features=input_features)
        == transformer.feature_names_in_
    )
    assert (
        transformer.get_feature_names_out(input_features=input_features)
        == variables_str
    )


@pytest.mark.parametrize("input_features", [None, variables_arr, np.array(variables_arr), variables_str, np.array(variables_str)])
def test_with_array(df_vartypes, input_features):
    # When the data used to train the class is a numpy array, the names stored in
    # feature_names_in_ are x0, x1, etc. Those should be returned by
    # get_feature_names_out() when input_features is None. Alternatively, it returns
    # a list of the variables entered by the user.
    transformer = MockClass()
    transformer.fit(df_vartypes.to_numpy())

    if input_features is None:
        assert (
            transformer.get_feature_names_out(input_features=input_features)
            == variables_arr
        )
    else:
        assert (
            transformer.get_feature_names_out(input_features=input_features)
            == list(input_features)
        )

@pytest.mark.parametrize("input_features", [None, variables_str, np.array(variables_str)])
def test_with_pipeline_and_df(df_vartypes, input_features):
    pipe = Pipeline([("transformer", MockClass())])
    pipe.fit(df_vartypes)
    assert (
        pipe.get_feature_names_out(input_features=input_features)
        == pipe.named_steps["transformer"].feature_names_in_
    )
    assert (
        pipe.get_feature_names_out(input_features=input_features)
        == variables_str
    )


@pytest.mark.parametrize("input_features", [None, variables_arr, np.array(variables_arr), variables_str, np.array(variables_str)])
def test_with_pipeline_and_array(df_vartypes, input_features):
    pipe = Pipeline([("transformer", MockClass())])
    pipe.fit(df_vartypes.to_numpy())

    if input_features is None:
        assert (
            pipe.get_feature_names_out(input_features=input_features)
            == variables_arr
        )
    else:
        assert (
            pipe.get_feature_names_out(input_features=input_features)
            == list(input_features)
        )

@pytest.mark.parametrize("input_features", [None, variables_str, np.array(variables_str)])
def test_with_pipe_and_skl_transformer_input_df(df_vartypes, input_features):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant")),
        ("transformer", MockClass()),
    ])
    pipe.fit(df_vartypes)
    assert (
        pipe.get_feature_names_out(input_features=input_features)
        == variables_str
    )

@pytest.mark.parametrize("input_features", [None, variables_arr, np.array(variables_arr), variables_str, np.array(variables_str)])
def test_with_pipe_and_skl_transformer_input_array(df_vartypes, input_features):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant")),
        ("transformer", MockClass()),
    ])
    pipe.fit(df_vartypes.to_numpy())

    if input_features is None:
        assert (
            pipe.get_feature_names_out(input_features=input_features)
            == variables_arr
        )
    else:
        assert (
            pipe.get_feature_names_out(input_features=input_features)
            == list(input_features)
        )

@pytest.mark.parametrize("input_features", [None, ["Age", "Marks"], np.array(["Age", "Marks"])])
def test_with_pipe_and_skl_transformer_that_adds_features(df_vartypes, input_features):
    pipe = Pipeline([
        ("poly", PolynomialFeatures()),
        ("transformer", MockClass()),
    ])

    # when input is array
    pipe.fit(df_vartypes[["Age", "Marks"]].to_numpy())
    assert (
        pipe.get_feature_names_out(input_features=None)
        == ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']
    )

    assert (
        pipe.get_feature_names_out(input_features=["Age", "Marks"])
        == ['1', 'Age', 'Marks', 'Age^2', 'Age Marks', 'Marks^2']
    )
    assert (
        pipe.get_feature_names_out(input_features=["Dog", "Cat"])
        == ['1', 'Dog', 'Cat', 'Dog^2', 'Dog Cat', 'Cat^2']
    )

    # when input is df
    pipe.fit(df_vartypes[["Age", "Marks"]])
    assert (
        pipe.get_feature_names_out(input_features=None)
        == ['1', 'Age', 'Marks', 'Age^2', 'Age Marks', 'Marks^2']
    )

    assert (
        pipe.get_feature_names_out(input_features=["Age", "Marks"])
        == ['1', 'Age', 'Marks', 'Age^2', 'Age Marks', 'Marks^2']
    )


def test_raise_error_when_input_feature_non_permitted(df_vartypes):
    transformer = MockClass()

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
