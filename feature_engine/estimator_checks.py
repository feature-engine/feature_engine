from typing import Tuple

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline


def test_df(
    categorical: bool = False, datetime: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates a dataframe that contains only numerical features, or additionally,
    categorical and datetime features.

    Parameters
    ----------
    categorical: bool, default=False
        Whether to add 2 additional categorical features.

    datetime: bool, default=False
        Whether to add one additional datetime feature.

    Returns
    -------
    X: pd.DataFrame
        A pandas dataframe
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform arrays into pandas df and series
    colnames = ["var_" + str(i) for i in range(12)]
    X = pd.DataFrame(X, columns=colnames)
    y = pd.Series(y)

    if categorical is True:
        X["cat_var1"] = ["A"] * 1000
        X["cat_var2"] = ["B"] * 1000

    if datetime is True:
        X["date1"] = pd.date_range("2020-02-24", periods=1000, freq="T")
        X["date2"] = pd.date_range("2021-09-29", periods=1000, freq="H")

    return X, y


def check_feature_engine_estimator(estimator):
    """
    Performs checks of common functionality to all transformers.

    There are checks that apply to all transformers, checks that run only when a
    parameter exists in the init method, and checks executed based on transformer
    tags. Tags are added to transformers to signal that some common tests should be
    performed on them.

    **Common tests:**

    - checks non-fitted error: checks error when transform() method is called before
    the fit() method.

    - checks that transformer raises error when input to fit() or transform() is not a
    dataframe.

    - checks correct values of attribute `features_names_in_`.

    - checks default functionality of method get_features_out().

    **Checks based on transformer's init attributes:**

    - checks that transformer can use any cross-validation constructor from sklearn.

    - checks correct functionality of parameter `drop_original`.

    - check that users enters permitted values to init parameters `missing_values`.

    **Checks based on transformer tags.**

    - checks that transformer raises error if y is not passed.

    - checks that numerical, categorical, datetime or all variables are correctly
    selected and assigned in fit().
    """
    # Tests for all transformers
    check_raises_non_fitted_error(estimator)
    check_raises_error_when_input_not_a_df(estimator)

    check_feature_names_in(estimator)
    check_get_feature_names_out(estimator)

    # Tests based on transformer tags
    tags = estimator._more_tags()

    if "requires_y" in tags.keys():
        check_error_if_y_not_passed(estimator)

    if hasattr(estimator, "variables"):
        if tags["variables"] == "numerical":
            check_numerical_variables_assignment(estimator)
        elif tags["variables"] == "categorical":
            check_categorical_variables_assignment(estimator)
        elif tags["variables"] == "all":
            check_all_types_variables_assignment(estimator)
        elif tags["variables"] == "datetime":
            check_datetime_variables_assignment(estimator)

    # Tests based on transformer's init parameters
    if hasattr(estimator, "cv"):
        check_takes_cv_constructor(estimator)

    if hasattr(estimator, "missing_values"):
        check_error_param_missing_values(estimator)

    if hasattr(estimator, "drop_original"):
        check_drop_original_variables(estimator)

    return None


# ======  Functionality shared by all transformers ======
def check_raises_non_fitted_error(estimator):
    """
    Check if transformer raises error when transform() method is called before
    calling fit() method.

    The functionality is provided by sklearn's `check_is_fitted` function.
    """
    X, y = test_df()
    transformer = clone(estimator)
    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        transformer.transform(X)


def check_raises_error_when_input_not_a_df(estimator):
    """
    Checks if transformer raises error when user passes input other than a pandas
    dataframe or numpy array to fit() or transform() methods.

    Functionality is provided by `is_dataframe`.
    """
    # non-permitted inputs.
    _not_a_df = [
        "not_a_df",
        [1, 2, 3, "some_data"],
        pd.Series([-2, 1.5, 8.94], name="not_a_df"),
    ]

    # permitted input
    X, y = test_df(categorical=True, datetime=True)

    transformer = clone(estimator)

    for not_df in _not_a_df:
        # test fitting not a df
        with pytest.raises(TypeError):
            transformer.fit(not_df)

        transformer.fit(X, y)
        # test transforming not a df
        with pytest.raises(TypeError):
            transformer.transform(not_df)


def check_feature_names_in(estimator):
    """Checks that all transformers learn the variable names of the train set used
    during fit."""
    # the estimator learns the parameters from the train set
    X, y = test_df(categorical=True, datetime=True)
    varnames = list(X.columns)
    estimator = clone(estimator)
    estimator.fit(X, y)
    assert estimator.feature_names_in_ == varnames
    assert estimator.n_features_in_ == len(varnames)


def check_get_feature_names_out(estimator):
    """
    Check that the method get_feature_names_out() returns the variable names of
    the transformed dataframe. In most transformers that would be the same as
    the variable names in the train set used in fit(). The value is stored in
    `feature_names_in_`.

    For those transformers that return additional variables, we need to incorporate
    specific tests, based on the transformer functionality. They will be skipped from
    this test.
    """
    _skip_test = [
        "OneHotEncoder",
        "AddMissingIndicator",
        "LagFeatures",
        "WindowFeatures",
        "ExpandingWindowFeatures",
        "MathFeatures",
        "CyclicalFeatures",
        "RelativeFeatures",
        "DatetimeFeatures",
    ]
    # the estimator learns the parameters from the train set
    X, y = test_df(categorical=True, datetime=True)
    estimator = clone(estimator)
    estimator.fit(X, y)

    # create Pipeline based on a transformer
    estimator2 = clone(estimator)
    pipe = Pipeline(["transformer", estimator2])
    pipe.fit(X, y)

    if estimator.__class__.__name__ not in _skip_test:
        # selection transformers
        if (
            hasattr(estimator, "confirm_variables")
            or estimator.__class__.__name__ == "DropFeatures"
        ):
            feature_names = [
                f for f in X.columns if f not in estimator.features_to_drop_
            ]
            assert estimator.get_feature_names_out() == feature_names
            assert estimator.transform(X).shape[1] == len(feature_names)

        else:
            # when 'input_features' is None, ie not specified
            assert estimator.get_feature_names_out() == [
                "var_" + str(i) for i in range(12)
            ] + [
                "cat_var1",
                "cat_var2",
                "date1",
                "date2",
            ]

            # user passes input features
            features = ["var_3", "var_5", "var_7", "cat_var2", "date1"]
            assert estimator.get_feature_names_out(
                input_features=features
            ) == features

            # transformer is used in a pipeline
            assert pipe.get_feature_names_out(
                input_features=features
            ) == features



# =======  TESTS BASED ON ESTIMATOR TAGS =============
def check_error_if_y_not_passed(estimator):
    """
    Checks that transformer raises error when y is not passed. Functionality is
    provided by Python, when making a parameter mandatory.

    For this test to run, we need to add the tag 'requires_y' to the transformer.
    """
    X, y = test_df()
    estimator = clone(estimator)
    with pytest.raises(TypeError):
        estimator.fit(X)


# ====== Check variable selection functionality ======
def check_numerical_variables_assignment(estimator):
    """
    Checks that transformers that work only with numerical variables, correctly set
    the values for the attributes `variables` and `variables_`.

    The first attribute can take a string, an integer, a list of strings or integers or
    None.

    The second attribute can take a list of string or integers, but is assigned after
    checking that the variables are of type numeric.

    For this check to run, the transformer needs the tag 'variables' set to 'numerical'.
    """
    # toy df
    X, y = test_df(categorical=True)

    # input variables to test
    _input_vars_ls = ["var_1", ["var_2"], ["var_1", "var_2", "var_3", "var_11"], None]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

        # before fitting
        if input_vars is not None:
            assert transformer.variables == input_vars
        else:
            assert transformer.variables is None

        # after fitting
        transformer.fit(X, y)

        if input_vars is not None:
            assert transformer.variables == input_vars

            if isinstance(input_vars, list):
                assert transformer.variables_ == input_vars
            else:
                assert transformer.variables_ == [input_vars]
        else:
            assert transformer.variables is None
            assert transformer.variables_ == ["var_" + str(i) for i in range(12)]

    # test raises error if uses passes categorical variable
    transformer.set_params(variables=["var_1", "cat_var1"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)


def check_categorical_variables_assignment(estimator):
    """
    Checks that transformers that work only with categorical variables, correctly set
    the values for the attributes `variables` and `variables_`.

    The first attribute can take a string, an integer, a list of strings or integers or
    None.

    The second attribute can take a list of string or integers, but is assigned after
    checking that the variables are of type object or categorical.

    For this check to run, the transformer needs the tag 'variables' set to
    'categorical'.
    """
    # toy df
    X, y = test_df(categorical=True)

    # cast one variable as category
    X[["cat_var2"]] = X[["cat_var2"]].astype("category")

    # input variables to test
    _input_vars_ls = ["cat_var1", ["cat_var1"], ["cat_var1", "cat_var2"], None]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

        # before fitting
        if input_vars is not None:
            assert transformer.variables == input_vars
        else:
            assert transformer.variables is None

        # fit
        transformer.fit(X, y)

        if input_vars is not None:
            assert transformer.variables == input_vars

            if isinstance(input_vars, list):
                assert transformer.variables_ == input_vars
            else:
                assert transformer.variables_ == [input_vars]
        else:
            assert transformer.variables is None
            assert transformer.variables_ == ["cat_var1", "cat_var2"]

    # test raises error if uses passes numerical variable
    transformer.set_params(variables=["var_1", "cat_var1"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)


def check_all_types_variables_assignment(estimator):
    """
    Checks that transformers that work with all types of variables, correctly set
    the values for the attributes `variables` and `variables_`.

    The first attribute can take a string, an integer, a list of strings or integers or
    None. The second attribute can take a list of string or integers.

    For this check to run, the transformer needs the tag 'variables' set to
    'all'.
    """

    # toy df
    X, y = test_df(categorical=True)

    # cast one variable as category
    X[["cat_var2"]] = X[["cat_var2"]].astype("category")

    # input variables to test
    _input_vars_ls = [
        "var_1",
        ["cat_var1"],
        ["var_1", "var_2", "cat_var1", "cat_var2"],
        None,
    ]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

        # before fitting
        if input_vars is not None:
            assert transformer.variables == input_vars
        else:
            assert transformer.variables is None

        # fit
        transformer.fit(X, y)

        if input_vars is not None:
            assert transformer.variables == input_vars

            if isinstance(input_vars, list):
                assert transformer.variables_ == input_vars
            else:
                assert transformer.variables_ == [input_vars]
        else:
            assert transformer.variables is None
            assert transformer.variables_ == list(X.columns)


def check_datetime_variables_assignment(estimator):
    """
    Checks that transformers that work with datetime variables, correctly set
    the values for the attributes `variables` and `variables_`.

    The first attribute can take a string, an integer, a list of strings or integers or
    None. The second attribute can take a list of string or integers.

    For this check to run, the transformer needs the tag 'variables' set to
    'datetime'.
    """

    # toy df
    X, y = test_df(datetime=True)

    # input variables to test
    _input_vars_ls = ["date1", ["date2"], ["date1", "date2"], None]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

        # before fitting
        if input_vars is not None:
            assert transformer.variables == input_vars
        else:
            assert transformer.variables is None

        # fit
        transformer.fit(X, y)

        if input_vars is not None:
            assert transformer.variables == input_vars

            if isinstance(input_vars, list):
                assert transformer.variables_ == input_vars
            else:
                assert transformer.variables_ == [input_vars]
        else:
            assert transformer.variables is None
            assert transformer.variables_ == ["date1", "date2"]

    # test raises error if uses passes numerical variable
    transformer.set_params(variables=["var_1", "date1"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)


# == TESTS BASED OF SPECIFIC PARAMETERS IN INIT SHARED CROSS TRANSFORMERS ===
def check_takes_cv_constructor(estimator):
    """
    Only for transformers with a parameter `cv`in init.

    For those transformers that implement cross-validation, checks that all
    sklearn cross-validation constructors can be used with the transformer.

    This checks corroborates that the attributes learned during fit() are indeed
    learned.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    X, y = test_df(categorical=True)

    estimator = clone(estimator)

    cv_constructor_ls = [KFold(n_splits=3), StratifiedKFold(n_splits=3), None]

    for cv_constructor in cv_constructor_ls:

        sel = estimator.set_params(cv=cv_constructor)
        sel.fit(X, y)
        Xtransformed = sel.transform(X)

        # test fit attrs
        if hasattr(sel, "initial_model_performance_"):
            assert isinstance(sel.initial_model_performance_, (int, float))

        if hasattr(sel, "features_to_drop_"):
            assert isinstance(sel.features_to_drop_, list)
            assert all([x for x in sel.features_to_drop_ if x in X.columns])
            assert len(sel.features_to_drop_) < X.shape[1]

            assert not Xtransformed.empty
            assert all(
                [x for x in Xtransformed.columns if x not in sel.features_to_drop_]
            )

        if hasattr(sel, "performance_drifts_"):
            assert isinstance(sel.performance_drifts_, dict)
            assert all([x for x in X.columns if x in sel.performance_drifts_.keys()])
            assert all(
                [
                    isinstance(sel.performance_drifts_[var], (int, float))
                    for var in sel.performance_drifts_.keys()
                ]
            )

        if hasattr(sel, "feature_performance_"):
            assert isinstance(sel.feature_performance_, dict)
            assert all([x for x in X.columns if x in sel.feature_performance_.keys()])
            assert all(
                [
                    isinstance(sel.feature_performance_[var], (int, float))
                    for var in sel.feature_performance_.keys()
                ]
            )

        if hasattr(sel, "scores_dict_"):
            assert isinstance(sel.scores_dict_, dict)
            assert all([x for x in X.columns if x in sel.scores_dict_.keys()])
            assert all(
                [
                    isinstance(sel.scores_dict_[var], (int, float))
                    for var in sel.scores_dict_.keys()
                ]
            )


def check_drop_original_variables(estimator):
    """
    Only for transformers with a parameter `drop_original`in init.

    Checks correct functionality of `drop_original`. If True, the original variables,
    that is, those stored in the attribute `variables_` are dropped from the
    transformed dataframe (after transform()). If False, original variables are
    returned in the transformed dataframe.
    """
    # Test df
    X, y = test_df(categorical=True, datetime=True)

    # when drop_original is true
    estimator = clone(estimator)
    estimator.set_params(drop_original=True)
    X_tr = estimator.fit_transform(X, y)

    if hasattr(estimator, "variables_"):
        vars = estimator.variables_
    elif hasattr(estimator, "reference"):
        vars = estimator.variables + estimator.reference
    else:
        vars = estimator.variables

    # Check that original variables are not in transformed dataframe
    assert set(vars).isdisjoint(set(X_tr.columns))
    # Check that remaining variables are in transformed dataframe
    remaining = [f for f in estimator.feature_names_in_ if f not in vars]
    assert all([f in X_tr.columns for f in remaining])

    # when drop_original is False
    estimator = clone(estimator)
    estimator.set_params(drop_original=False)
    X_tr = estimator.fit_transform(X, y)

    if hasattr(estimator, "variables_"):
        vars = estimator.variables_
    else:
        vars = estimator.variables

    # Check that original variables are in transformed dataframe
    assert len([f in X_tr.columns for f in vars])
    # Check that remaining variables are in transformed dataframe
    remaining = [f for f in estimator.feature_names_in_ if f not in vars]
    assert all([f in X_tr.columns for f in remaining])


def check_error_param_missing_values(estimator):
    """
    Only for transformers with a parameter `missing_values`in init.

    Checks transformer raises error when user enters non-permitted value to the
    parameter.
    """
    # param takes values "raise" or "ignore"
    estimator = clone(estimator)
    for value in [2, "hola", False]:
        if estimator.__class__.__name__ == "MathFeatures":
            with pytest.raises(ValueError):
                estimator.__class__(
                    variables=["var_1", "var_2", "var_3"],
                    func="mean",
                    missing_values=value,
                )

        elif estimator.__class__.__name__ == "RelativeFeatures":
            with pytest.raises(ValueError):
                estimator.__class__(
                    variables=["var_1", "var_2", "var_3"],
                    reference=["var_4"],
                    func="mean",
                    missing_values=value,
                )
        else:
            with pytest.raises(ValueError):
                estimator.__class__(missing_values=value)


def check_confirm_variables(estimator):
    """
    Only for transformers with a parameter `confirm_variables`in init.

    At the moment, this test applies to variable selection transformers. The idea is
    to corroborate if the variables entered by the user are present in the dataframe
    before doing the selection, when the parameter is True.
    """
    X, y = test_df()
    Xs = X.drop(labels=["var_10", "var_11"], axis=1)

    # original variables in X
    all_vars = ["var_" + str(i) for i in range(12)]

    estimator = clone(estimator)

    sel = estimator.set_params(
        variables=all_vars,
        confirm_variables=False,
    )
    sel.fit(X, y)
    assert sel.variables_ == all_vars

    sel = estimator.set_params(
        variables=all_vars,
        confirm_variables=True,
    )
    sel.fit(Xs, y)
    assert sel.variables_ == ["var_" + str(i) for i in range(10)]

    sel = estimator.set_params(
        variables=all_vars,
        confirm_variables=False,
    )
    with pytest.raises(KeyError):
        sel.fit(Xs, y)

    # When variables is None.
    sel = estimator.set_params(
        variables=None,
        confirm_variables=True,
    )
    sel.fit(X, y)
    assert sel.variables_ == all_vars

    sel.fit(Xs, y)
    assert sel.variables_ == ["var_" + str(i) for i in range(10)]
