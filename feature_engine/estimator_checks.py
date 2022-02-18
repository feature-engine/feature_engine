import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError


def test_df(numeric=True, datetime=False):
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # trasform arrays into pandas df and series
    colnames = ["var_" + str(i) for i in range(12)]
    X = pd.DataFrame(X, columns=colnames)
    y = pd.Series(y)

    if numeric is False:
        X["cat_var"] = ["A"] * 1000
        X["cat_var2"] = ["B"] * 1000

    if datetime is True:
        X["date"] = pd.date_range("2020-02-24", periods=1000, freq="T")

    return X, y


def check_feature_engine_estimator(estimator):
    # Gather all tests
    check_raises_non_fitted_error(estimator)
    check_raises_error_when_fitting_not_a_df(estimator)
    check_raises_error_when_transforming_not_a_df(estimator)

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

    if hasattr(estimator, "cv"):
        check_takes_cv_constructor(estimator)

    if hasattr(estimator, "missing_values"):
        check_error_param_missing_values(estimator)

    check_feature_names_in(estimator)
    check_get_feature_names_out(estimator)

    if hasattr(estimator, "drop_original"):
        check_drop_original_variables(estimator)

# ======  Common setting errors ======
def check_raises_non_fitted_error(estimator):
    X, y = test_df()
    transformer = clone(estimator)
    # test when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer.transform(X)


def check_raises_error_when_fitting_not_a_df(estimator):
    _not_a_df = [
        "not_a_df",
        [1, 2, 3, "some_data"],
        pd.Series([-2, 1.5, 8.94], name="not_a_df"),
    ]

    transformer = clone(estimator)
    for not_df in _not_a_df:
        # trying to fit not a df
        with pytest.raises(TypeError):
            transformer.fit(not_df)


def check_raises_error_when_transforming_not_a_df(estimator):
    X, y = test_df(numeric=False)

    _not_a_df = [
        "not_a_df",
        [1, 2, 3, "some_data"],
        pd.Series([-2, 1.5, 8.94], name="not_a_df"),
    ]

    transformer = clone(estimator)
    transformer.fit(X, y)

    for not_df in _not_a_df:
        # trying to transform not a df
        with pytest.raises(TypeError):
            transformer.fit(not_df)


def check_error_if_y_not_passed(estimator):
    X, y = test_df()
    estimator = clone(estimator)
    with pytest.raises(TypeError):
        estimator.fit(X)


# ====== Check variable selection functionality ======
def check_numerical_variables_assignment(estimator):
    # toy df
    X, y = test_df(numeric=False)

    # input variables to test
    _input_vars_ls = ["var_1", ["var_2"], ["var_1", "var_2", "var_3", "var_11"], None]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

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
            assert transformer.variables_ == ["var_" + str(i) for i in range(12)]

    # test raises error if uses passes categorical variable
    transformer.set_params(variables=["var_1", "cat_var"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)


def check_categorical_variables_assignment(estimator):
    # toy df
    X, y = test_df(numeric=False)

    # cast one variable as category
    X[["cat_var2"]] = X[["cat_var2"]].astype("category")

    # input variables to test
    _input_vars_ls = ["cat_var", ["cat_var"], ["cat_var", "cat_var2"], None]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

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
            assert transformer.variables_ == ["cat_var", "cat_var2"]

    # test raises error if uses passes numerical variable
    transformer.set_params(variables=["var_1", "cat_var"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)


def check_all_types_variables_assignment(estimator):
    # toy df
    X, y = test_df(numeric=False)

    # cast one variable as category
    X[["cat_var2"]] = X[["cat_var2"]].astype("category")

    # input variables to test
    _input_vars_ls = [
        "var_1",
        ["cat_var"],
        ["var_1", "var_2", "cat_var", "cat_var2"],
        None,
    ]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

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


# ====== Check functionality shared across transformers ======
def check_takes_cv_constructor(estimator):
    from sklearn.model_selection import KFold, StratifiedKFold

    X, y = test_df(numeric=False)

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
    X, y = test_df(numeric=False, datetime=True)
    estimator = clone(estimator)
    estimator.set_params(drop_original=True)
    X_tr = estimator.fit_transform(X, y)
    # Check that original variables are not in transformed dataframe
    assert len([f for f in X_tr.columns if f in estimator.variables_]) == 0
    # Check that remaining variables are in transformed dataframe
    remaining = [
        f for f in estimator.feature_names_in_ if f not in estimator.variables_
    ]
    assert all([f in X_tr.columns for f in remaining])


# ======== Check common fit attributes ========
def check_feature_names_in(estimator):
    # the estimator learns the parameters from the train set
    X, y = test_df(numeric=False, datetime=True)
    estimator = clone(estimator)
    estimator.fit(X, y)
    assert estimator.feature_names_in_ == ["var_" + str(i) for i in range(12)] + [
        "cat_var",
        "cat_var2",
        "date",
    ]


# ======== Check common methods ========
def check_get_feature_names_out(estimator):
    # the estimator learns the parameters from the train set
    X, y = test_df(numeric=False, datetime=True)
    estimator = clone(estimator)
    estimator.fit(X, y)

    if estimator.__class__.__name__ in ["OneHotEncoder"]:
        pass

    # selection transformers
    elif (
        hasattr(estimator, "confirm_variables")
        or estimator.__class__.__name__ == "DropFeatures"
    ):
        feature_names = [f for f in X.columns if f not in estimator.features_to_drop_]
        assert estimator.get_feature_names_out() == feature_names
        assert estimator.transform(X).shape[1] == len(feature_names)

    else:
        assert estimator.get_feature_names_out() == [
            "var_" + str(i) for i in range(12)
        ] + [
            "cat_var",
            "cat_var2",
            "date",
        ]


# ======== Check errors when non-permitted input params ======
def check_error_param_missing_values(estimator):
    # param takes values "raise" or "ignore"
    estimator = clone(estimator)
    for value in [2, "hola", False]:
        with pytest.raises(ValueError):
            estimator.__class__(missing_values=value)


def check_confirm_variables(estimator):
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
