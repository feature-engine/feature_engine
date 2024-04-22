import numpy as np
import pandas as pd
import pytest
from sklearn import __version__ as skl_version
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_regression,
)
from sklearn.impute import MissingIndicator, SimpleImputer
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

from feature_engine.wrappers import SklearnTransformerWrapper

_transformers = [
    Binarizer(threshold=2),
    KBinsDiscretizer(n_bins=3, encode="ordinal"),
    StandardScaler(),
    MinMaxScaler(),
    Normalizer(),
    PowerTransformer(),
    FunctionTransformer(np.log, validate=True),
    OrdinalEncoder(),
]

_selectors = [
    SelectFromModel(Lasso(random_state=1)),
    SelectKBest(f_regression, k=2),
    VarianceThreshold(),
    RFE(Lasso(random_state=1)),
]


def _OneHotEncoder(sparse, drop=None, dtype=np.float64) -> OneHotEncoder:
    """OneHotEncoder sparse argument has been renamed as sparse_output
    in scikitlearn >=1.2"""

    if skl_version.split(".")[0] == "1" and int(skl_version.split(".")[1]) >= 2:
        return OneHotEncoder(sparse_output=sparse, drop=drop, dtype=dtype)
    else:
        return OneHotEncoder(sparse=sparse, drop=drop, dtype=dtype)


@pytest.mark.parametrize(
    "transformer",
    [
        SimpleImputer(),
        _OneHotEncoder(sparse=False),
        StandardScaler(),
        SelectKBest(),
    ],
)
def test_permitted_param_transformer(transformer, df_na):
    tr = SklearnTransformerWrapper(transformer=transformer)
    assert tr.transformer == transformer


@pytest.mark.parametrize("transformer", [Lasso(), RandomForestClassifier()])
def test_error_when_transformer_is_estimator(transformer, df_na):
    with pytest.raises(TypeError):
        SklearnTransformerWrapper(transformer=transformer)


@pytest.mark.parametrize(
    "transformer",
    [
        PCA(),
        VotingClassifier(RandomForestClassifier()),
        MissingIndicator(),
        KBinsDiscretizer(encode="one_hot"),
        SimpleImputer(add_indicator=True),
        _OneHotEncoder(sparse=True),
    ],
)
def test_error_not_implemented_transformer(transformer, df_na):
    with pytest.raises(NotImplementedError):
        SklearnTransformerWrapper(transformer=transformer)


@pytest.mark.parametrize("transformer", _selectors)
def test_wrap_selectors(transformer):
    # load data
    X = fetch_california_housing(as_frame=True).frame
    y = X["MedHouseVal"]
    X = X.drop(["MedHouseVal"], axis=1)

    # prepare selectors
    sel = clone(transformer)
    sel_wrap = SklearnTransformerWrapper(transformer=transformer)

    # Test:
    # When passing variable list
    varlist = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
    sel_wrap.set_params(variables=varlist)

    Xt = pd.DataFrame(
        sel.fit_transform(X[varlist], y),
        columns=X[varlist].columns[(sel.get_support())],
    )
    Xw = sel_wrap.fit_transform(X, y)

    selected = X[varlist].columns[(sel.get_support())]
    remaining = [f for f in X.columns if f not in varlist]

    pd.testing.assert_frame_equal(Xt, Xw[selected])
    pd.testing.assert_frame_equal(X[remaining], Xw[remaining])
    assert Xw.shape[1] == len(remaining) + len(selected)

    # when variable list is None
    sel_wrap.set_params(variables=None)

    Xt = pd.DataFrame(sel.fit_transform(X, y), columns=X.columns[(sel.get_support())])
    Xw = sel_wrap.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xt, Xw)


@pytest.mark.parametrize("transformer", _transformers)
def test_wrap_transformers(transformer):
    # load data
    X = fetch_california_housing(as_frame=True).frame

    # prepare selectors
    tr = clone(transformer)
    tr_wrap = SklearnTransformerWrapper(transformer=transformer)

    # Test:
    # When passing variable list
    varlist = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
    tr_wrap.set_params(variables=varlist)

    Xt = pd.DataFrame(tr.fit_transform(X[varlist]), columns=X[varlist].columns)
    Xw = tr_wrap.fit_transform(X)

    remaining = [f for f in X.columns if f not in varlist]

    assert Xt.shape[1] == 4
    assert Xw.shape[1] == 9
    pd.testing.assert_frame_equal(Xt, Xw[varlist])
    pd.testing.assert_frame_equal(X[remaining], Xw[remaining])

    # when variable list is None
    tr_wrap.set_params(variables=None)

    Xt = pd.DataFrame(tr.fit_transform(X), columns=X.columns)
    Xw = tr_wrap.fit_transform(X)

    pd.testing.assert_frame_equal(Xt, Xw)


def test_wrap_polynomial_features():
    # load data
    X = fetch_california_housing(as_frame=True).frame

    # prepare selectors
    tr = PolynomialFeatures()
    tr_wrap = SklearnTransformerWrapper(transformer=PolynomialFeatures())

    # Test:
    # When passing variable list
    varlist = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
    tr_wrap.set_params(variables=varlist)

    Xt = pd.DataFrame(
        tr.fit_transform(X[varlist]), columns=tr.get_feature_names_out(varlist)
    )
    Xw = tr_wrap.fit_transform(X)

    pd.testing.assert_frame_equal(Xw, pd.concat([X.drop(columns=varlist), Xt], axis=1))
    assert Xw.shape[1] == len(X.drop(columns=varlist).columns) + len(
        tr.get_feature_names_out(varlist)
    )

    # when variable list is None
    tr_wrap.set_params(variables=None)

    Xt = pd.DataFrame(tr.fit_transform(X), columns=tr.get_feature_names_out())
    Xw = tr_wrap.fit_transform(X)

    pd.testing.assert_frame_equal(Xw, Xt)
    assert Xw.shape[1] == len(tr.get_feature_names_out(X.columns))


def test_wrap_polynomial_features_get_features_name_out():
    X = fetch_california_housing(as_frame=True).frame

    varlist = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
    tr_wrap = SklearnTransformerWrapper(
        transformer=PolynomialFeatures(), variables=varlist
    )

    tr_wrap.fit(X)
    expected_features_all = [
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedHouseVal",
        "1",
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "MedInc^2",
        "MedInc HouseAge",
        "MedInc AveRooms",
        "MedInc AveBedrms",
        "HouseAge^2",
        "HouseAge AveRooms",
        "HouseAge AveBedrms",
        "AveRooms^2",
        "AveRooms AveBedrms",
        "AveBedrms^2",
    ]
    expected_features_varlist = [
        "1",
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "MedInc^2",
        "MedInc HouseAge",
        "MedInc AveRooms",
        "MedInc AveBedrms",
        "HouseAge^2",
        "HouseAge AveRooms",
        "HouseAge AveBedrms",
        "AveRooms^2",
        "AveRooms AveBedrms",
        "AveBedrms^2",
    ]

    assert tr_wrap.get_feature_names_out() == expected_features_all
    assert tr_wrap.get_feature_names_out(varlist) == expected_features_varlist


# SimpleImputer
def test_wrap_simple_imputer(df_na):
    variables_to_impute = ["Age", "Marks"]
    na_variables_left_after_imputation = [
        col
        for col in df_na.loc[:, df_na.isna().any()].columns
        if col not in variables_to_impute
    ]

    transformer = SklearnTransformerWrapper(
        transformer=SimpleImputer(fill_value=-999, strategy="constant"),
        variables=variables_to_impute,
    )

    # transformed dataframe
    ref = df_na.copy()
    ref[variables_to_impute] = ref[variables_to_impute].fillna(-999)

    dataframe_na_transformed = transformer.fit_transform(df_na)

    # transformed output
    assert all(
        dataframe_na_transformed[na_variables_left_after_imputation].isna().sum() != 0
    )
    assert all(dataframe_na_transformed[variables_to_impute].isna().sum() == 0)
    pd.testing.assert_frame_equal(ref, dataframe_na_transformed)


def test_sklearn_imputer_object_with_constant(df_na):
    variables_to_impute = ["Name", "City"]
    na_variables_left_after_imputation = [
        col
        for col in df_na.loc[:, df_na.isna().any()].columns
        if col not in variables_to_impute
    ]

    transformer = SklearnTransformerWrapper(
        transformer=SimpleImputer(fill_value="missing", strategy="constant"),
        variables=variables_to_impute,
    )

    # transformed dataframe
    ref = df_na.copy()
    ref[variables_to_impute] = ref[variables_to_impute].fillna("missing")

    dataframe_na_transformed = transformer.fit_transform(df_na)

    # transformed output
    assert all(
        dataframe_na_transformed[na_variables_left_after_imputation].isna().sum() != 0
    )
    assert all(dataframe_na_transformed[variables_to_impute].isna().sum() == 0)
    pd.testing.assert_frame_equal(ref, dataframe_na_transformed)


def test_sklearn_imputer_allfeatures_with_constant(df_na):
    transformer = SklearnTransformerWrapper(
        transformer=SimpleImputer(fill_value="missing", strategy="constant")
    )

    # transformed dataframe
    ref = df_na.copy()
    ref = ref.fillna("missing")

    dataframe_na_transformed = transformer.fit_transform(df_na)

    # transformed output
    assert all(dataframe_na_transformed.isna().sum() == 0)
    pd.testing.assert_frame_equal(ref, dataframe_na_transformed)


# One Hot Encoder
def test_sklearn_ohe_object_one_feature(df_vartypes):
    variables_to_encode = ["Name"]

    transformer = SklearnTransformerWrapper(
        transformer=_OneHotEncoder(sparse=False, dtype=np.int64),
        variables=variables_to_encode,
    )

    ref = pd.DataFrame(
        {
            "Name_jack": [0, 0, 0, 1],
            "Name_krish": [0, 0, 1, 0],
            "Name_nick": [0, 1, 0, 0],
            "Name_tom": [1, 0, 0, 0],
        }
    )

    transformed_df = transformer.fit_transform(df_vartypes[variables_to_encode])

    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_object_many_features(df_vartypes):
    variables_to_encode = ["Name", "City"]

    transformer = SklearnTransformerWrapper(
        transformer=_OneHotEncoder(sparse=False, dtype=np.int64),
        variables=variables_to_encode,
    )

    ref = pd.DataFrame(
        {
            "Name_jack": [0, 0, 0, 1],
            "Name_krish": [0, 0, 1, 0],
            "Name_nick": [0, 1, 0, 0],
            "Name_tom": [1, 0, 0, 0],
            "City_Bristol": [0, 0, 0, 1],
            "City_Liverpool": [0, 0, 1, 0],
            "City_London": [1, 0, 0, 0],
            "City_Manchester": [0, 1, 0, 0],
        }
    )

    transformed_df = transformer.fit_transform(df_vartypes[variables_to_encode])

    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_numeric(df_vartypes):
    variables_to_encode = ["Age"]

    transformer = SklearnTransformerWrapper(
        transformer=_OneHotEncoder(sparse=False, dtype=np.int64),
        variables=variables_to_encode,
    )

    ref = pd.DataFrame(
        {
            "Age_18": [0, 0, 0, 1],
            "Age_19": [0, 0, 1, 0],
            "Age_20": [1, 0, 0, 0],
            "Age_21": [0, 1, 0, 0],
        }
    )

    transformed_df = transformer.fit_transform(df_vartypes[variables_to_encode])

    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_all_features(df_vartypes):
    transformer = SklearnTransformerWrapper(
        transformer=_OneHotEncoder(sparse=False, dtype=np.int64)
    )

    ref = pd.DataFrame(
        {
            "Name_jack": [0, 0, 0, 1],
            "Name_krish": [0, 0, 1, 0],
            "Name_nick": [0, 1, 0, 0],
            "Name_tom": [1, 0, 0, 0],
            "City_Bristol": [0, 0, 0, 1],
            "City_Liverpool": [0, 0, 1, 0],
            "City_London": [1, 0, 0, 0],
            "City_Manchester": [0, 1, 0, 0],
            "Age_18": [0, 0, 0, 1],
            "Age_19": [0, 0, 1, 0],
            "Age_20": [1, 0, 0, 0],
            "Age_21": [0, 1, 0, 0],
            "Marks_0.6": [0, 0, 0, 1],
            "Marks_0.7": [0, 0, 1, 0],
            "Marks_0.8": [0, 1, 0, 0],
            "Marks_0.9": [1, 0, 0, 0],
            "dob_2020-02-24T00:00:00.000000000": [1, 0, 0, 0],
            "dob_2020-02-24T00:01:00.000000000": [0, 1, 0, 0],
            "dob_2020-02-24T00:02:00.000000000": [0, 0, 1, 0],
            "dob_2020-02-24T00:03:00.000000000": [0, 0, 0, 1],
        }
    )

    transformed_df = transformer.fit_transform(df_vartypes)

    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_with_crossvalidation():
    """
    Created 2022-02-14 to test fix to issue # 368
    """

    # Set up test pipeline with wrapped OneHotEncoder, with simple regression model
    # to be able to run cross-validation; use sklearn CA housing data
    df = fetch_california_housing(as_frame=True).frame
    y = df["MedHouseVal"]
    X = (
        df[["HouseAge", "AveBedrms"]]
        .assign(
            AveBedrms_cat=lambda x: pd.cut(x.AveBedrms, [0, 1, 2, 3, 4, np.inf]).astype(
                str
            )
        )
        .drop(columns="AveBedrms")
    )
    pipeline: Pipeline = Pipeline(
        steps=[
            (
                "encode_cat",
                SklearnTransformerWrapper(
                    transformer=_OneHotEncoder(drop="first", sparse=False),
                    variables=["AveBedrms_cat"],
                ),
            ),
            ("model", Lasso()),
        ]
    )

    # Run cross-validation
    results: np.ndarray = cross_val_score(
        pipeline, X, y, scoring="neg_mean_squared_error", cv=3
    )
    assert not any([np.isnan(i) for i in results])


def test_wrap_one_hot_encoder_get_features_name_out(df_vartypes):
    ohe_wrap = SklearnTransformerWrapper(transformer=_OneHotEncoder(sparse=False))
    ohe_wrap.fit(df_vartypes)

    expected_features_all = [
        "Name_jack",
        "Name_krish",
        "Name_nick",
        "Name_tom",
        "City_Bristol",
        "City_Liverpool",
        "City_London",
        "City_Manchester",
        "Age_18",
        "Age_19",
        "Age_20",
        "Age_21",
        "Marks_0.6",
        "Marks_0.7",
        "Marks_0.8",
        "Marks_0.9",
        "dob_2020-02-24T00:00:00.000000000",
        "dob_2020-02-24T00:01:00.000000000",
        "dob_2020-02-24T00:02:00.000000000",
        "dob_2020-02-24T00:03:00.000000000",
    ]

    assert ohe_wrap.get_feature_names_out() == expected_features_all


@pytest.mark.parametrize(
    "transformer",
    [PowerTransformer(), OrdinalEncoder(), MinMaxScaler(), StandardScaler()],
)
def test_inverse_transform(transformer):
    X = fetch_california_housing(as_frame=True).frame
    X = X.drop(["Longitude"], axis=1)

    tr_wrap = SklearnTransformerWrapper(transformer=transformer)

    # When passing variable list
    varlist = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
    tr_wrap.set_params(variables=varlist)
    X_tr = tr_wrap.fit_transform(X)
    X_inv = tr_wrap.inverse_transform(X_tr)

    pd.testing.assert_frame_equal(X_inv, X)

    # when variable list is None
    tr_wrap.set_params(variables=None)

    X_tr = tr_wrap.fit_transform(X)
    X_inv = tr_wrap.inverse_transform(X_tr)

    pd.testing.assert_frame_equal(X_inv, X)


@pytest.mark.parametrize(
    "transformer",
    [
        SelectKBest(f_regression, k=2),
        PolynomialFeatures(),
        SimpleImputer(),
    ],
)
def test_error_when_inverse_transform_not_implemented(transformer):
    X = fetch_california_housing(as_frame=True).frame
    y = X["MedHouseVal"]
    X = X.drop(["MedHouseVal"], axis=1)

    tr_wrap = SklearnTransformerWrapper(transformer=transformer)
    tr_wrap.fit(X, y)
    X_tr = tr_wrap.transform(X)

    with pytest.raises(NotImplementedError):
        tr_wrap.inverse_transform(X_tr)


@pytest.mark.parametrize(
    "varlist", [["MedInc", "HouseAge", "AveRooms", "AveBedrms"], None]
)
@pytest.mark.parametrize("transformer", _transformers)
def test_get_feature_names_out_transformers(varlist, transformer):
    X = fetch_california_housing(as_frame=True).frame
    tr_wrap = SklearnTransformerWrapper(transformer=transformer, variables=varlist)
    Xw = tr_wrap.fit_transform(X)

    assert Xw.columns.to_list() == tr_wrap.get_feature_names_out()
    assert Xw.columns.to_list() == tr_wrap.get_feature_names_out(["MedInc", "HouseAge"])


@pytest.mark.parametrize(
    "varlist", [["MedInc", "HouseAge", "AveRooms", "AveBedrms"], None]
)
@pytest.mark.parametrize("transformer", _selectors)
def test_get_feature_names_out_selectors(varlist, transformer):
    X = fetch_california_housing(as_frame=True).frame
    y = X["MedHouseVal"]
    X = X.drop(["MedHouseVal"], axis=1)
    tr_wrap = SklearnTransformerWrapper(transformer=transformer, variables=varlist)
    Xw = tr_wrap.fit_transform(X, y)

    assert Xw.columns.to_list() == tr_wrap.get_feature_names_out()
    assert Xw.columns.to_list() == tr_wrap.get_feature_names_out(["MedInc", "HouseAge"])


@pytest.mark.parametrize(
    "varlist", [["MedInc", "HouseAge", "AveRooms", "AveBedrms"], None]
)
def test_get_feature_names_out_polynomialfeatures(varlist):
    X = fetch_california_housing(as_frame=True).frame
    tr_wrap = SklearnTransformerWrapper(
        transformer=PolynomialFeatures(), variables=varlist
    )
    Xw = tr_wrap.fit_transform(X)
    assert Xw.columns.tolist() == tr_wrap.get_feature_names_out()

    if varlist is not None:
        output_feat = [
            "1",
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "MedInc^2",
            "MedInc HouseAge",
            "MedInc AveRooms",
            "MedInc AveBedrms",
            "HouseAge^2",
            "HouseAge AveRooms",
            "HouseAge AveBedrms",
            "AveRooms^2",
            "AveRooms AveBedrms",
            "AveBedrms^2",
        ]

        assert output_feat == tr_wrap.get_feature_names_out(varlist)


@pytest.mark.parametrize("varlist", [["Name", "City"], None])
def test_get_feature_names_out_ohe(varlist, df_vartypes):
    transformer = SklearnTransformerWrapper(
        transformer=_OneHotEncoder(sparse=False, dtype=np.int64),
        variables=varlist,
    )

    df_tr = transformer.fit_transform(df_vartypes)

    assert df_tr.columns.to_list() == transformer.get_feature_names_out()

    if varlist is not None:
        output_feat = [
            "Name_jack",
            "Name_krish",
            "Name_nick",
            "Name_tom",
            "City_Bristol",
            "City_Liverpool",
            "City_London",
            "City_Manchester",
        ]

        assert output_feat == transformer.get_feature_names_out(varlist)


def test_function_transformer_works_with_categoricals():
    X = pd.DataFrame({"col1": ["1", "2", "3"], "col2": ["a", "b", "c"]})

    X_expected = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": ["a", "b", "c"]})

    transformer = SklearnTransformerWrapper(
        FunctionTransformer(lambda x: x.astype(np.float64)), variables=["col1"]
    )

    X_tf = transformer.fit_transform(X)

    pd.testing.assert_frame_equal(X_expected, X_tf)


def test_function_transformer_works_with_numericals():
    X = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    X_expected = pd.DataFrame({"col1": [2, 3, 4], "col2": ["a", "b", "c"]})

    transformer = SklearnTransformerWrapper(
        FunctionTransformer(lambda x: x + 1), variables=["col1"]
    )

    X_tf = transformer.fit_transform(X)

    pd.testing.assert_frame_equal(X_expected, X_tf)
