import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engine.wrappers import SklearnTransformerWrapper


def test_sklearn_imputer_numeric_with_constant(df_na):
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

    # init params
    assert isinstance(transformer.transformer, SimpleImputer)
    assert transformer.variables == variables_to_impute
    # fit params
    assert transformer.variables_ == variables_to_impute
    assert transformer.n_features_in_ == 6
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

    # init params
    assert isinstance(transformer.transformer, SimpleImputer)
    assert transformer.variables == variables_to_impute
    # fit params
    assert transformer.n_features_in_ == 6
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

    # init params
    assert isinstance(transformer.transformer, SimpleImputer)
    # fit params
    assert transformer.n_features_in_ == 6
    # transformed output
    assert all(dataframe_na_transformed.isna().sum() == 0)
    pd.testing.assert_frame_equal(ref, dataframe_na_transformed)


def test_sklearn_standardscaler_numeric(df_vartypes):
    variables_to_scale = ["Age", "Marks"]
    transformer = SklearnTransformerWrapper(
        transformer=StandardScaler(), variables=variables_to_scale
    )

    ref = df_vartypes.copy()
    ref[variables_to_scale] = (
        ref[variables_to_scale] - ref[variables_to_scale].mean()
    ) / ref[variables_to_scale].std(ddof=0)

    transformed_df = transformer.fit_transform(df_vartypes)

    # init params
    assert isinstance(transformer.transformer, StandardScaler)
    assert transformer.variables == variables_to_scale
    # fit params
    assert transformer.n_features_in_ == 5
    assert (transformer.transformer_.mean_.round(6) == np.array([19.5, 0.75])).all()
    assert all(transformer.transformer_.scale_.round(6) == [1.118034, 0.111803])
    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_standardscaler_object(df_vartypes):
    variables_to_scale = ["Name"]
    transformer = SklearnTransformerWrapper(
        transformer=StandardScaler(), variables=variables_to_scale
    )

    with pytest.raises(TypeError):
        transformer.fit_transform(df_vartypes)

    # init params
    assert isinstance(transformer.transformer, StandardScaler)
    assert transformer.variables == variables_to_scale


def test_sklearn_standardscaler_allfeatures(df_vartypes):
    transformer = SklearnTransformerWrapper(transformer=StandardScaler())

    ref = df_vartypes.copy()
    variables_to_scale = list(ref.select_dtypes(include="number").columns)
    ref[variables_to_scale] = (
        ref[variables_to_scale] - ref[variables_to_scale].mean()
    ) / ref[variables_to_scale].std(ddof=0)

    transformed_df = transformer.fit_transform(df_vartypes)

    # init params
    assert isinstance(transformer.transformer, StandardScaler)
    assert transformer.variables is None
    # fit params
    assert transformer.variables_ == variables_to_scale
    assert transformer.n_features_in_ == 5
    assert (transformer.transformer_.mean_.round(6) == np.array([19.5, 0.75])).all()
    assert all(transformer.transformer_.scale_.round(6) == [1.118034, 0.111803])
    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_object_one_feature(df_vartypes):
    variables_to_encode = ["Name"]

    transformer = SklearnTransformerWrapper(
        transformer=OneHotEncoder(sparse=False, dtype=np.int64),
        variables=variables_to_encode,
    )

    ref = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "Name_jack": [0, 0, 0, 1],
            "Name_krish": [0, 0, 1, 0],
            "Name_nick": [0, 1, 0, 0],
            "Name_tom": [1, 0, 0, 0],
        }
    )

    transformed_df = transformer.fit_transform(df_vartypes[variables_to_encode])

    # init params
    assert isinstance(transformer.transformer, OneHotEncoder)
    assert transformer.variables == variables_to_encode
    # fit params
    assert transformer.n_features_in_ == 1
    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_object_many_features(df_vartypes):
    variables_to_encode = ["Name", "City"]

    transformer = SklearnTransformerWrapper(
        transformer=OneHotEncoder(sparse=False, dtype=np.int64),
        variables=variables_to_encode,
    )

    ref = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
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

    # init params
    assert isinstance(transformer.transformer, OneHotEncoder)
    assert transformer.variables == variables_to_encode
    # fit params
    assert transformer.n_features_in_ == 2
    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_numeric(df_vartypes):
    variables_to_encode = ["Age"]

    transformer = SklearnTransformerWrapper(
        transformer=OneHotEncoder(sparse=False, dtype=np.int64),
        variables=variables_to_encode,
    )

    ref = pd.DataFrame(
        {
            "Age": [20, 21, 19, 18],
            "Age_18": [0, 0, 0, 1],
            "Age_19": [0, 0, 1, 0],
            "Age_20": [1, 0, 0, 0],
            "Age_21": [0, 1, 0, 0],
        }
    )

    transformed_df = transformer.fit_transform(df_vartypes[variables_to_encode])

    # init params
    assert isinstance(transformer.transformer, OneHotEncoder)
    assert transformer.variables == variables_to_encode
    # fit params
    assert transformer.n_features_in_ == 1
    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_all_features(df_vartypes):
    transformer = SklearnTransformerWrapper(
        transformer=OneHotEncoder(sparse=False, dtype=np.int64)
    )

    ref = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
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

    # init params
    assert isinstance(transformer.transformer, OneHotEncoder)
    # fit params
    assert transformer.n_features_in_ == 5
    pd.testing.assert_frame_equal(ref, transformed_df)


def test_sklearn_ohe_errors(df_vartypes):
    with pytest.raises(AttributeError):
        SklearnTransformerWrapper(transformer=OneHotEncoder(sparse=True)).fit(
            df_vartypes
        )


def test_selectKBest_all_variables():
    X, y = load_boston(return_X_y=True)
    X = pd.DataFrame(X)

    selector = SklearnTransformerWrapper(
        transformer=SelectKBest(f_regression, k=5),
    )

    selector.fit(X, y)

    X_train_t = selector.transform(X)

    pd.testing.assert_frame_equal(X_train_t, X[[2, 5, 9, 10, 12]])


def test_selectFromModel_all_variables():
    X, y = load_boston(return_X_y=True)
    X = pd.DataFrame(X)

    lasso = Lasso(alpha=10, random_state=0)

    sfm = SelectFromModel(lasso, prefit=False)

    selector = SklearnTransformerWrapper(transformer=sfm)

    selector.fit(X, y)

    X_train_t = selector.transform(X)

    pd.testing.assert_frame_equal(X_train_t, X[[1, 9, 11, 12]])


def test_selectFromModel_selected_variables():
    X, y = load_boston(return_X_y=True)
    X = pd.DataFrame(X)

    lasso = Lasso(alpha=10, random_state=0)

    sfm = SelectFromModel(lasso, prefit=False)

    selector = SklearnTransformerWrapper(
        transformer=sfm,
        variables=[0, 1, 2, 3, 4, 5],
    )

    selector.fit(X, y)

    X_train_t = selector.transform(X)

    pd.testing.assert_frame_equal(X_train_t, X[[0, 1, 2, 6, 7, 8, 9, 10, 11, 12]])


def test_raise_error_if_transformer_wrong_type():
    with pytest.raises(TypeError):
        SklearnTransformerWrapper(
            transformer="hola",
            variables=[0, 1, 2, 3, 4, 5],
        )
