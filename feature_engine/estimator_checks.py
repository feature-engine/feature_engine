import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError


def test_df(numeric=True):
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

    return X, y


def check_feature_engine_estimator(estimator):
    # TODO: test if this is working
    check_raises_non_fitted_error(estimator)
    check_raises_error_when_fitting_not_a_df
    check_raises_error_when_transforming_not_a_df(estimator)


def check_raises_non_fitted_error(estimator):
    X, y = test_df()
    transformer = estimator
    # test when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer.transform(X)


def check_raises_error_when_fitting_not_a_df(estimator):
    _not_a_df = [
        "not_a_df",
        [1, 2, 3, "some_data"],
        pd.Series([-2, 1.5, 8.94], name="not_a_df"),
    ]

    transformer = estimator
    for not_df in _not_a_df:
        # trying to fit not a df
        with pytest.raises(TypeError):
            transformer.fit(not_df)


def check_raises_error_when_transforming_not_a_df(estimator):
    X, y = test_df()

    _not_a_df = [
        "not_a_df",
        [1, 2, 3, "some_data"],
        pd.Series([-2, 1.5, 8.94], name="not_a_df"),
    ]

    transformer = estimator
    transformer.fit(X, y)

    for not_df in _not_a_df:
        # trying to transform not a df
        with pytest.raises(TypeError):
            transformer.fit(not_df)


def check_numerical_variables_assignment(estimator):
    # toy df
    X, y = test_df(numeric=False)

    # input variables to test
    _input_vars_ls = ["var_1", ["var_2"], ["var_1", "var_2", "var_3", "var_11"], None]

    # the estimator
    transformer = estimator

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
    transformer = estimator

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
    transformer = estimator

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
