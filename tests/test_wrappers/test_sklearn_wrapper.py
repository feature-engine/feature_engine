import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn import config_context
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    StandardScaler,
)

from feature_engine.wrappers import SklearnTransformerWrapper, SklearnWrapper
from tests.backend_helpers import frame_to_dict, make_series, null_count

DATA = {
    "num_1": [1, 2, 3, 4],
    "num_2": [2.0, 4.0, 6.0, 8.0],
    "cat": ["a", "b", "a", "c"],
}

DATA_NA = {
    "num": [1.0, None, 3.0, 5.0],
    "cat": ["a", None, "a", "b"],
    "other": [1, 2, 3, 4],
}

# x1 follows the target, x2 does not
DATA_SELECTION = {
    "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "x2": [6.0, 1.0, 5.0, 2.0, 4.0, 3.0],
    "cat": ["a", "b", "a", "b", "a", "b"],
}
TARGET = [1.1, 2.0, 2.9, 4.2, 5.0, 6.1]

# StandardScaler of [1, 2, 3, 4], or of any linear transformation of it
SCALED = [
    -1.3416407864998738,
    -0.4472135954999579,
    0.4472135954999579,
    1.3416407864998738,
]

NOT_FITTED_MSG = (
    "This SklearnWrapper instance is not fitted yet. Call 'fit' with "
    "appropriate arguments before using this estimator."
)


# init parameters
@pytest.mark.parametrize(
    "transformer", [Lasso(), RandomForestClassifier(), "StandardScaler", 1, None]
)
def test_error_if_transformer_is_not_a_transformer(transformer):
    msg = (
        "transformer expected a Scikit-learn transformer. "
        f"Got {transformer} instead."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        SklearnWrapper(transformer=transformer)


@pytest.mark.parametrize(
    "transformer",
    [PCA(), VotingClassifier([("rf", RandomForestClassifier())]), MissingIndicator()],
)
def test_error_if_transformer_is_not_supported(transformer):
    msg = (
        "This transformer is not compatible with the wrapper. Supported "
        "transformers are GenericUnivariateSelect, RFE, RFECV, SelectFdr, "
        "SelectFpr, SelectFromModel, SelectFwe, SelectKBest, SelectPercentile, "
        "SequentialFeatureSelector, VarianceThreshold, OneHotEncoder, "
        "PolynomialFeatures, Binarizer, FunctionTransformer, KBinsDiscretizer, "
        "PowerTransformer, QuantileTransformer, SimpleImputer, IterativeImputer, "
        "KNNImputer, OrdinalEncoder, MaxAbsScaler, MinMaxScaler, StandardScaler, "
        "RobustScaler, Normalizer."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        SklearnWrapper(transformer=transformer)


@pytest.mark.parametrize(
    "transformer",
    [
        SimpleImputer(add_indicator=True),
        KNNImputer(add_indicator=True),
        IterativeImputer(add_indicator=True),
    ],
)
def test_error_if_imputer_adds_indicator(transformer):
    msg = (
        "The imputer is only compatible with the wrapper when the "
        "parameter `add_indicator` is False. "
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        SklearnWrapper(transformer=transformer)


@pytest.mark.parametrize("encode", ["onehot", "onehot-dense"])
def test_error_if_kbins_discretizer_encoding_is_not_ordinal(encode):
    msg = (
        "The KBinsDiscretizer is only compatible with the wrapper when the "
        "parameter `encode` is `ordinal`. "
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        SklearnWrapper(transformer=KBinsDiscretizer(encode=encode))


def test_error_if_one_hot_encoder_output_is_sparse():
    msg = "SklearnWrapper can only wrap OneHotEncoder if the sparse is set to False."
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        SklearnWrapper(transformer=OneHotEncoder(sparse_output=True))


@pytest.mark.parametrize(
    "transformer",
    [
        SimpleImputer(),
        OneHotEncoder(sparse_output=False),
        StandardScaler(),
        SelectKBest(),
        KBinsDiscretizer(encode="ordinal"),
    ],
)
def test_init_param_assignment(transformer):
    wrapper = SklearnWrapper(transformer=transformer)
    assert wrapper.transformer is transformer


def test_sklearn_transformer_wrapper_is_deprecated(make_df):
    msg = (
        "SklearnTransformerWrapper was deprecated in favour of SklearnWrapper in "
        "version 2.0.0 and will be removed in version 2.1.0. To silence this "
        "warning, use SklearnWrapper instead."
    )
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        wrapper = SklearnTransformerWrapper(
            transformer=StandardScaler(), variables=["num_1"]
        )
    assert isinstance(wrapper, SklearnWrapper)

    Xt = wrapper.fit_transform(make_df(DATA))
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {**DATA, "num_1": pytest.approx(SCALED)}


# fit and transform
@pytest.mark.parametrize(
    "transformer, expected",
    [
        (StandardScaler(), {"num_1": SCALED, "num_2": SCALED}),
        (
            MinMaxScaler(),
            {"num_1": [0.0, 1 / 3, 2 / 3, 1.0], "num_2": [0.0, 1 / 3, 2 / 3, 1.0]},
        ),
        (Binarizer(threshold=2), {"num_1": [0, 0, 1, 1], "num_2": [0, 1, 1, 1]}),
        (
            FunctionTransformer(np.cbrt, validate=True),
            {
                "num_1": [1.0, 1.2599210498948732, 1.4422495703074083, 1.5874010519681],
                "num_2": [
                    1.2599210498948732,
                    1.5874010519681,
                    1.8171205928321397,
                    2.0,
                ],
            },
        ),
    ],
)
def test_transformers_replace_the_variables(make_df, transformer, expected):
    wrapper = SklearnWrapper(transformer=transformer, variables=["num_1", "num_2"])
    Xt = wrapper.fit_transform(make_df(DATA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "num_1": pytest.approx(expected["num_1"]),
        "num_2": pytest.approx(expected["num_2"]),
        "cat": ["a", "b", "a", "c"],
    }


def test_variables_none_selects_numerical_variables(make_df):
    wrapper = SklearnWrapper(transformer=StandardScaler())
    Xt = wrapper.fit_transform(make_df(DATA))

    assert wrapper.variables_ == ["num_1", "num_2"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "num_1": pytest.approx(SCALED),
        "num_2": pytest.approx(SCALED),
        "cat": ["a", "b", "a", "c"],
    }


def test_variables_none_selects_all_variables_with_ordinal_encoder(make_df):
    wrapper = SklearnWrapper(transformer=OrdinalEncoder())
    Xt = wrapper.fit_transform(make_df(DATA))

    assert wrapper.variables_ == ["num_1", "num_2", "cat"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "num_1": [0.0, 1.0, 2.0, 3.0],
        "num_2": [0.0, 1.0, 2.0, 3.0],
        "cat": [0.0, 1.0, 0.0, 2.0],
    }


def test_transform_returns_variables_in_the_order_seen_in_fit(make_df):
    wrapper = SklearnWrapper(transformer=StandardScaler(), variables=["num_1"])
    wrapper.fit(make_df(DATA))
    Xt = wrapper.transform(make_df({k: DATA[k] for k in ["cat", "num_2", "num_1"]}))

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["num_1", "num_2", "cat"]
    assert frame_to_dict(Xt) == {**DATA, "num_1": pytest.approx(SCALED)}


def test_simple_imputer_with_numerical_variables(make_df):
    wrapper = SklearnWrapper(transformer=SimpleImputer(), variables=["num"])
    Xt = wrapper.fit_transform(make_df(DATA_NA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {**DATA_NA, "num": [1.0, 3.0, 3.0, 5.0]}


@pytest.mark.parametrize(
    "transformer, expected",
    [
        (SimpleImputer(strategy="most_frequent"), ["a", "a", "a", "b"]),
        (
            SimpleImputer(strategy="constant", fill_value="missing"),
            ["a", "missing", "a", "b"],
        ),
    ],
)
def test_simple_imputer_with_categorical_variables(make_df, transformer, expected):
    wrapper = SklearnWrapper(transformer=transformer, variables=["cat"])
    Xt = wrapper.fit_transform(make_df(DATA_NA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {**DATA_NA, "cat": expected}


def test_missing_values_stay_missing(make_df):
    wrapper = SklearnWrapper(transformer=StandardScaler(), variables=["num"])
    Xt = wrapper.fit_transform(make_df(DATA_NA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA_NA,
        "num": [pytest.approx(-1.224744871391589), None, 0.0, 1.224744871391589],
    }
    assert null_count(Xt, "num") == 1


def test_ordinal_encoder_leaves_missing_values_missing(make_df):
    wrapper = SklearnWrapper(transformer=OrdinalEncoder(), variables=["cat"])
    Xt = wrapper.fit_transform(make_df(DATA_NA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {**DATA_NA, "cat": [0.0, None, 0.0, 1.0]}
    assert null_count(Xt, "cat") == 1


@pytest.mark.parametrize(
    "transformer, expected",
    [
        (
            OneHotEncoder(sparse_output=False),
            {
                "cat_a": [1.0, 0.0, 1.0, 0.0],
                "cat_b": [0.0, 1.0, 0.0, 0.0],
                "cat_c": [0.0, 0.0, 0.0, 1.0],
            },
        ),
        (
            OneHotEncoder(sparse_output=False, drop="first", dtype=np.int64),
            {"cat_b": [0, 1, 0, 0], "cat_c": [0, 0, 0, 1]},
        ),
    ],
)
def test_one_hot_encoder_adds_variables_at_the_end(make_df, transformer, expected):
    wrapper = SklearnWrapper(transformer=transformer, variables="cat")
    Xt = wrapper.fit_transform(make_df(DATA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "num_1": [1, 2, 3, 4],
        "num_2": [2.0, 4.0, 6.0, 8.0],
        **expected,
    }


def test_one_hot_encoder_encodes_missing_values_as_a_category(make_df):
    wrapper = SklearnWrapper(
        transformer=OneHotEncoder(sparse_output=False), variables=["cat"]
    )
    Xt = wrapper.fit_transform(make_df(DATA_NA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "num": [1.0, None, 3.0, 5.0],
        "other": [1, 2, 3, 4],
        "cat_a": [1.0, 0.0, 1.0, 0.0],
        "cat_b": [0.0, 0.0, 0.0, 1.0],
        "cat_nan": [0.0, 1.0, 0.0, 0.0],
    }


@pytest.mark.parametrize(
    "include_bias, expected",
    [
        (
            True,
            {
                "cat": ["a", "b", "a", "c"],
                "1": [1.0, 1.0, 1.0, 1.0],
                "num_1": [1.0, 2.0, 3.0, 4.0],
                "num_2": [2.0, 4.0, 6.0, 8.0],
                "num_1^2": [1.0, 4.0, 9.0, 16.0],
                "num_1 num_2": [2.0, 8.0, 18.0, 32.0],
                "num_2^2": [4.0, 16.0, 36.0, 64.0],
            },
        ),
        (
            False,
            {
                "cat": ["a", "b", "a", "c"],
                "num_1": [1.0, 2.0, 3.0, 4.0],
                "num_2": [2.0, 4.0, 6.0, 8.0],
                "num_1^2": [1.0, 4.0, 9.0, 16.0],
                "num_1 num_2": [2.0, 8.0, 18.0, 32.0],
                "num_2^2": [4.0, 16.0, 36.0, 64.0],
            },
        ),
    ],
)
def test_polynomial_features_adds_variables_at_the_end(make_df, include_bias, expected):
    wrapper = SklearnWrapper(transformer=PolynomialFeatures(include_bias=include_bias))
    Xt = wrapper.fit_transform(make_df(DATA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == expected
    assert wrapper.get_feature_names_out() == list(expected.keys())


def test_polynomial_features_when_all_variables_are_transformed(make_df):
    wrapper = SklearnWrapper(transformer=PolynomialFeatures(include_bias=False))
    Xt = wrapper.fit_transform(make_df({"num_1": [1, 2, 3, 4]}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "num_1": [1.0, 2.0, 3.0, 4.0],
        "num_1^2": [1.0, 4.0, 9.0, 16.0],
    }


def test_variance_threshold_drops_variables(make_df):
    wrapper = SklearnWrapper(transformer=VarianceThreshold(threshold=2))
    Xt = wrapper.fit_transform(make_df(DATA))

    assert wrapper.features_to_drop_ == ["num_1"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"num_2": DATA["num_2"], "cat": DATA["cat"]}


def test_select_k_best_uses_the_target(make_df):
    wrapper = SklearnWrapper(transformer=SelectKBest(f_regression, k=1))
    Xt = wrapper.fit_transform(make_df(DATA_SELECTION), make_series(make_df, TARGET))

    assert wrapper.features_to_drop_ == ["x2"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "x1": DATA_SELECTION["x1"],
        "cat": DATA_SELECTION["cat"],
    }


@pytest.mark.parametrize("y", [TARGET, np.array(TARGET)])
def test_target_as_list_or_array(make_df, y):
    wrapper = SklearnWrapper(transformer=SelectKBest(f_regression, k=1))
    Xt = wrapper.fit_transform(make_df(DATA_SELECTION), y)

    assert wrapper.features_to_drop_ == ["x2"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "x1": DATA_SELECTION["x1"],
        "cat": DATA_SELECTION["cat"],
    }


def test_function_transformer_receives_the_input_dataframe(make_df):
    def to_float(X):
        # the function gets the user's dataframe, pandas or polars
        return nw.from_native(X).with_columns(nw.all().cast(nw.Float64)).to_native()

    wrapper = SklearnWrapper(
        transformer=FunctionTransformer(to_float), variables=["col1"]
    )
    Xt = wrapper.fit_transform(make_df({"col1": ["1", "2", "3"], "col2": [1, 2, 3]}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"col1": [1.0, 2.0, 3.0], "col2": [1, 2, 3]}


def test_function_transformer_with_numerical_variables(make_df):
    wrapper = SklearnWrapper(
        transformer=FunctionTransformer(lambda x: x + 1), variables=["col1"]
    )
    Xt = wrapper.fit_transform(make_df({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"col1": [2, 3, 4], "col2": ["a", "b", "c"]}


@pytest.mark.parametrize(
    "transform_output, output_type",
    [("pandas", pd.DataFrame), ("polars", pl.DataFrame)],
)
def test_sklearn_transform_output_config(make_df, transform_output, output_type):
    with config_context(transform_output=transform_output):
        Xt = SklearnWrapper(
            transformer=OneHotEncoder(sparse_output=False), variables=["cat"]
        ).fit_transform(make_df(DATA))

    assert isinstance(Xt, output_type)
    assert frame_to_dict(Xt) == {
        "num_1": [1, 2, 3, 4],
        "num_2": [2.0, 4.0, 6.0, 8.0],
        "cat_a": [1.0, 0.0, 1.0, 0.0],
        "cat_b": [0.0, 1.0, 0.0, 0.0],
        "cat_c": [0.0, 0.0, 0.0, 1.0],
    }


def test_wrapper_in_pipeline_with_cross_validation(make_df):
    # regression test for issue #368
    rng = np.random.RandomState(0)
    X = make_df(
        {
            "num": rng.normal(size=30).tolist(),
            "cat": rng.choice(["a", "b", "c"], size=30).tolist(),
        }
    )
    y = make_series(make_df, rng.normal(size=30).tolist())
    pipeline = Pipeline(
        steps=[
            (
                "encoder",
                SklearnWrapper(
                    transformer=OneHotEncoder(sparse_output=False, drop="first"),
                    variables=["cat"],
                ),
            ),
            ("model", Lasso()),
        ]
    )

    results = cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=3)
    assert np.isfinite(results).all()


@pytest.mark.parametrize(
    "transformer",
    [StandardScaler(), MinMaxScaler(), PowerTransformer(), OrdinalEncoder()],
)
def test_inverse_transform(make_df, transformer):
    wrapper = SklearnWrapper(transformer=transformer, variables=["num_1", "num_2"])
    Xt = wrapper.fit_transform(make_df(DATA))
    Xt_before = frame_to_dict(Xt)
    X_inv = wrapper.inverse_transform(Xt)

    assert isinstance(X_inv, make_df)
    assert frame_to_dict(X_inv) == {
        "num_1": pytest.approx(DATA["num_1"]),
        "num_2": pytest.approx(DATA["num_2"]),
        "cat": DATA["cat"],
    }
    # the input dataframe is not modified
    assert frame_to_dict(Xt) == Xt_before


def test_inverse_transform_with_ordinal_encoder_and_categorical_variables(make_df):
    wrapper = SklearnWrapper(transformer=OrdinalEncoder(), variables=["cat"])
    X_inv = wrapper.inverse_transform(wrapper.fit_transform(make_df(DATA)))

    assert isinstance(X_inv, make_df)
    assert frame_to_dict(X_inv) == DATA


@pytest.mark.parametrize(
    "transformer",
    [
        SelectKBest(f_regression, k=1),
        PolynomialFeatures(),
        SimpleImputer(strategy="most_frequent"),
    ],
)
def test_error_if_inverse_transform_is_not_supported(make_df, transformer):
    msg = (
        "The method `inverse_transform` is not implemented for this transformer. "
        "Supported transformers are PowerTransformer, QuantileTransformer, "
        "OrdinalEncoder, MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler."
    )
    X = make_df(DATA_SELECTION)
    wrapper = SklearnWrapper(transformer=transformer).fit(
        X, make_series(make_df, TARGET)
    )

    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        wrapper.inverse_transform(wrapper.transform(X))


@pytest.mark.parametrize("variables", [["num_1", "num_2"], None])
@pytest.mark.parametrize(
    "transformer",
    [
        StandardScaler(),
        Binarizer(threshold=2),
        Normalizer(),
        MinMaxScaler(),
    ],
)
def test_get_feature_names_out_transformers(make_df, transformer, variables):
    wrapper = SklearnWrapper(transformer=transformer, variables=variables)
    Xt = wrapper.fit_transform(make_df(DATA))

    assert wrapper.get_feature_names_out() == ["num_1", "num_2", "cat"]
    assert wrapper.get_feature_names_out() == list(Xt.columns)
    # input_features is ignored
    assert wrapper.get_feature_names_out(["num_1"]) == ["num_1", "num_2", "cat"]


@pytest.mark.parametrize("variables", [["x1", "x2"], None])
def test_get_feature_names_out_selectors(make_df, variables):
    wrapper = SklearnWrapper(
        transformer=SelectKBest(f_regression, k=1), variables=variables
    )
    Xt = wrapper.fit_transform(make_df(DATA_SELECTION), make_series(make_df, TARGET))

    assert wrapper.get_feature_names_out() == ["x1", "cat"]
    assert wrapper.get_feature_names_out() == list(Xt.columns)
    # input_features is ignored
    assert wrapper.get_feature_names_out(["x1"]) == ["x1", "cat"]


def test_get_feature_names_out_one_hot_encoder(make_df):
    wrapper = SklearnWrapper(
        transformer=OneHotEncoder(sparse_output=False), variables=["cat", "num_1"]
    )
    Xt = wrapper.fit_transform(make_df(DATA))
    new_features = [
        "cat_a",
        "cat_b",
        "cat_c",
        "num_1_1",
        "num_1_2",
        "num_1_3",
        "num_1_4",
    ]

    assert wrapper.get_feature_names_out() == ["num_2"] + new_features
    assert wrapper.get_feature_names_out() == list(Xt.columns)
    assert wrapper.get_feature_names_out(["cat", "num_1"]) == new_features


def test_get_feature_names_out_polynomial_features(make_df):
    wrapper = SklearnWrapper(
        transformer=PolynomialFeatures(), variables=["num_1", "num_2"]
    )
    Xt = wrapper.fit_transform(make_df(DATA))
    new_features = ["1", "num_1", "num_2", "num_1^2", "num_1 num_2", "num_2^2"]

    assert wrapper.get_feature_names_out() == ["cat"] + new_features
    assert wrapper.get_feature_names_out() == list(Xt.columns)
    assert wrapper.get_feature_names_out(["num_1", "num_2"]) == new_features


def test_return_empty_when_no_numerical_variables(make_df):
    msg = "No numerical variables found in this dataframe. Returning an empty list."
    X = make_df({"cat": ["a", "b", "a"]})
    wrapper = SklearnWrapper(transformer=StandardScaler(), return_empty=True)

    with pytest.warns(UserWarning, match=re.escape(msg)):
        wrapper.fit(X)
    Xt = wrapper.transform(X)

    assert wrapper.variables_ == []
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"cat": ["a", "b", "a"]}
    assert wrapper.get_feature_names_out() == ["cat"]


def test_error_if_no_numerical_variables_and_return_empty_is_false(make_df):
    msg = (
        "No numerical variables found in this dataframe. Check variable dtypes "
        "or set return_empty to True to return an empty list instead."
    )
    wrapper = SklearnWrapper(transformer=StandardScaler())
    with pytest.raises(TypeError, match=re.escape(msg)):
        wrapper.fit(make_df({"cat": ["a", "b", "a"]}))


def test_error_if_transform_df_has_different_number_of_columns(make_df):
    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer (when using the fit() method)."
    )
    wrapper = SklearnWrapper(transformer=StandardScaler()).fit(make_df(DATA))
    with pytest.raises(ValueError, match=re.escape(msg)):
        wrapper.transform(make_df({"num_1": DATA["num_1"]}))


@pytest.mark.parametrize(
    "method", ["transform", "inverse_transform", "get_feature_names_out"]
)
def test_error_if_not_fitted(make_df, method):
    wrapper = SklearnWrapper(transformer=StandardScaler())
    with pytest.raises(NotFittedError, match=re.escape(NOT_FITTED_MSG)):
        if method == "get_feature_names_out":
            wrapper.get_feature_names_out()
        else:
            getattr(wrapper, method)(make_df(DATA))


# pandas-specific behaviour
def test_pandas_integer_column_names():
    X = pd.DataFrame(
        {0: [1, 2, 3, 4], 1: [2.0, 4.0, 6.0, 8.0], 2: ["a", "b", "a", "c"]}
    )

    wrapper = SklearnWrapper(transformer=StandardScaler())
    Xt = wrapper.fit_transform(X)
    expected = pd.DataFrame({0: SCALED, 1: SCALED, 2: ["a", "b", "a", "c"]})
    pd.testing.assert_frame_equal(Xt, expected)
    pd.testing.assert_frame_equal(wrapper.inverse_transform(Xt), X, check_dtype=False)

    wrapper = SklearnWrapper(transformer=VarianceThreshold(threshold=2))
    pd.testing.assert_frame_equal(wrapper.fit_transform(X), X[[1, 2]])


@pytest.mark.parametrize(
    "transformer, expected",
    [
        (StandardScaler(), {"num_1": SCALED, "num_2": SCALED, "cat": DATA["cat"]}),
        (
            OneHotEncoder(sparse_output=False),
            {
                "num_1": DATA["num_1"],
                "num_2": DATA["num_2"],
                "cat_a": [1.0, 0.0, 1.0, 0.0],
                "cat_b": [0.0, 1.0, 0.0, 0.0],
                "cat_c": [0.0, 0.0, 0.0, 1.0],
            },
        ),
        (VarianceThreshold(threshold=2), {"num_2": DATA["num_2"], "cat": DATA["cat"]}),
    ],
)
def test_pandas_index_is_kept(transformer, expected):
    index = [10, 10, 5, 7]
    variables = ["cat"] if isinstance(transformer, OneHotEncoder) else None
    wrapper = SklearnWrapper(transformer=transformer, variables=variables)
    Xt = wrapper.fit_transform(pd.DataFrame(DATA, index=index))

    pd.testing.assert_frame_equal(Xt, pd.DataFrame(expected, index=index))


def test_pandas_one_hot_encoder_with_category_dtype():
    X = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"]), "num": [1, 2, 3]})
    wrapper = SklearnWrapper(
        transformer=OneHotEncoder(sparse_output=False), variables=["cat"]
    )
    expected = pd.DataFrame(
        {"num": [1, 2, 3], "cat_a": [1.0, 0.0, 1.0], "cat_b": [0.0, 1.0, 0.0]}
    )
    pd.testing.assert_frame_equal(wrapper.fit_transform(X), expected)
