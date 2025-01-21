import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine._prediction.base_predictor import BaseTargetMeanEstimator
from feature_engine._prediction.target_mean_classifier import TargetMeanClassifier
from feature_engine._prediction.target_mean_regressor import TargetMeanRegressor
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)
from feature_engine.encoding import MeanEncoder
from tests.estimator_checks.dataframe_for_checks import test_df
from tests.estimator_checks.fit_functionality_checks import check_error_if_y_not_passed

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

_estimators = [BaseTargetMeanEstimator(), TargetMeanClassifier(), TargetMeanRegressor()]
_predictors = [TargetMeanRegressor(), TargetMeanClassifier()]

if sklearn_version < parse_version("1.6"):
    # In sklearn version 1.6, changes into the developer api were introduced
    # that break the tests. Need to dig further into it.
    # TODO: add tests for sklearn version > 1.6
    @pytest.mark.parametrize("estimator", [BaseTargetMeanEstimator()])
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_error_if_y_not_passed(estimator):
    return check_error_if_y_not_passed(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_raises_non_fitted_error(df_vartypes, estimator):
    transformer = clone(estimator)
    if hasattr(transformer, "_predict"):
        with pytest.raises(NotFittedError):
            transformer._predict(df_vartypes)
    else:
        with pytest.raises(NotFittedError):
            transformer.predict(df_vartypes)


@pytest.mark.parametrize("estimator", _estimators)
def test_raises_error_when_input_not_a_df(estimator):
    # non-permitted inputs.
    _not_a_df = [
        "not_a_df",
        [1, 2, 3, "some_data"],
        pd.Series([-2, 1.5, 8.94], name="not_a_df"),
    ]

    transformer = clone(estimator)

    # Error in fit param:
    for not_df in _not_a_df:
        # test fitting not a df
        with pytest.raises(TypeError):
            transformer.fit(not_df)

    # error in transform param:
    X, y = test_df(categorical=True, datetime=True)

    if transformer.__class__.__name__ == "TargetMeanRegressor":
        # y needs to be continuous
        y = X["var_1"]
        X.drop(["var_1"], axis=1, inplace=True)

    transformer.fit(X, y)
    for not_df in _not_a_df:
        if hasattr(transformer, "_predict"):
            with pytest.raises(TypeError):
                transformer._predict(not_df)
        else:
            with pytest.raises(TypeError):
                transformer.predict(not_df)


_false_input_params = [
    ("salsa", "arbitrary"),
    ("33", "mean-encoder"),
    ([7], True),
    (0.2, "prost"),
]


@pytest.mark.parametrize("estimator", _estimators)
@pytest.mark.parametrize("_bins, _strategy", _false_input_params)
def test_raises_error_when_wrong_input_params(_bins, _strategy, estimator):
    transformer = clone(estimator)
    with pytest.raises(ValueError):
        assert transformer.__class__(bins=_bins)
    with pytest.raises(ValueError):
        assert transformer.__class__(strategy=_strategy)


@pytest.mark.parametrize("estimator", _estimators)
def test_variable_selection(estimator):

    transformer = clone(estimator)

    X, y = test_df(categorical=True, datetime=True)

    if transformer.__class__.__name__ == "TargetMeanRegressor":
        # y needs to be continuous
        y = X["var_10"]
        X.drop(["var_10"], axis=1, inplace=True)

    # cast one variable as category
    X[["cat_var2"]] = X[["cat_var2"]].astype("category")

    # cast datetime as object
    X[["date1"]] = X[["date1"]].astype("O")

    # Case 1: numerical variable as string
    transformer.set_params(variables="var_1")
    assert transformer.variables == "var_1"

    transformer.fit(X, y)
    assert transformer.variables == "var_1"
    assert transformer.variables_categorical_ == []
    assert transformer.variables_numerical_ == ["var_1"]

    # Case 2: numerical variable as list
    transformer.set_params(variables=["var_1"])
    assert transformer.variables == ["var_1"]

    transformer.fit(X, y)
    assert transformer.variables == ["var_1"]
    assert transformer.variables_categorical_ == []
    assert transformer.variables_numerical_ == ["var_1"]

    # Case 3: categorical variable as string
    transformer.set_params(variables="cat_var1")
    assert transformer.variables == "cat_var1"

    transformer.fit(X, y)
    assert transformer.variables == "cat_var1"
    assert transformer.variables_categorical_ == ["cat_var1"]
    assert transformer.variables_numerical_ == []

    # Case 4: categorical variable as list
    transformer.set_params(variables=["cat_var1"])
    assert transformer.variables == ["cat_var1"]

    transformer.fit(X, y)
    assert transformer.variables == ["cat_var1"]
    assert transformer.variables_categorical_ == ["cat_var1"]
    assert transformer.variables_numerical_ == []

    # Case 5: numerical and categorical variables
    transformer.set_params(
        variables=["var_1", "var_2", "cat_var1", "cat_var2", "date1"]
    )
    assert transformer.variables == ["var_1", "var_2", "cat_var1", "cat_var2", "date1"]

    transformer.fit(X, y)
    assert transformer.variables == ["var_1", "var_2", "cat_var1", "cat_var2", "date1"]
    assert transformer.variables_categorical_ == ["cat_var1", "cat_var2", "date1"]
    assert transformer.variables_numerical_ == ["var_1", "var_2"]

    # Case 6: automatically select variables
    X_c = X[["var_1", "var_2", "cat_var1", "cat_var2", "date1", "date2"]].copy()

    transformer.set_params(variables=None)
    assert transformer.variables is None

    transformer.fit(X_c, y)
    assert transformer.variables is None
    assert transformer.variables_categorical_ == ["cat_var1", "cat_var2"]
    assert transformer.variables_numerical_ == ["var_1", "var_2"]

    transformer.set_params(variables=["var_1", "cat_var1", "date2"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)

    # Case 6: user passes empty list
    transformer.set_params(variables=[])
    with pytest.raises(ValueError):
        transformer.fit(X, y)


@pytest.mark.parametrize("estimator", _estimators)
def test_feature_names_in(estimator):

    transformer = clone(estimator)
    X, y = test_df(categorical=True)

    if transformer.__class__.__name__ == "TargetMeanRegressor":
        # y needs to be continuous
        y = X["var_10"]

    varnames = list(X.columns)

    transformer.fit(X, y)

    assert transformer.feature_names_in_ == varnames
    assert transformer.n_features_in_ == len(varnames)


@pytest.mark.parametrize("_strategy", ["equal_width", "equal_frequency"])
@pytest.mark.parametrize("_bins", [3, 5, 7])
@pytest.mark.parametrize("estimator", _estimators)
def test_attributes_upon_fitting(_strategy, _bins, estimator):
    transformer = clone(estimator)
    transformer.set_params(bins=_bins, strategy=_strategy)

    X, y = test_df(categorical=True)

    if transformer.__class__.__name__ == "TargetMeanRegressor":
        # y needs to be continuous
        y = X["var_10"]

    transformer.fit(X, y)

    assert transformer.bins == _bins
    assert transformer.strategy == _strategy

    if _strategy == "equal_width":
        assert (
            type(transformer._pipeline.named_steps["discretiser"])
            is EqualWidthDiscretiser
        )
    else:
        assert (
            type(transformer._pipeline.named_steps["discretiser"])
            is EqualFrequencyDiscretiser
        )

    assert type(transformer._pipeline.named_steps["encoder_num"]) is MeanEncoder

    assert type(transformer._pipeline.named_steps["encoder_cat"]) is MeanEncoder


@pytest.mark.parametrize("estimator", _estimators)
def test_raises_error_when_df_has_nan(df_enc, df_na, estimator):

    transformer = clone(estimator)

    X, y = test_df(categorical=True)
    X_na = X.copy()

    X_na.loc[0, "var_1"] = np.nan

    if transformer.__class__.__name__ != "TargetMeanRegressor":
        # Raise error when dataset contains na, fit method
        with pytest.raises(ValueError):
            transformer.fit(X_na, y)

        transformer.fit(X, y)
        if hasattr(transformer, "_predict"):
            with pytest.raises(ValueError):
                transformer._predict(X_na)
        else:
            with pytest.raises(ValueError):
                transformer.predict(X_na)
            with pytest.raises(ValueError):
                transformer.predict_proba(X_na)
            with pytest.raises(ValueError):
                transformer.predict_log_proba(X_na)

    else:
        y = X["var_10"]
        with pytest.raises(ValueError):
            transformer.fit(X_na, y)

        transformer.fit(X, y)
        with pytest.raises(ValueError):
            transformer.predict(X_na)
