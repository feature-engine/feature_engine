"""Many transformers have init parameters that trigger similar functionality, like
checking for missing values, allowing cross-validation, dropping original variables,
etc.

In this script, we add common tests for the functionality triggered by those
parameters.
"""
import pytest
from sklearn import clone

from tests.estimator_checks.dataframe_for_checks import test_df


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

    vars = estimator.variables_
    if hasattr(estimator, "reference"):
        vars = vars + estimator.reference

    # Check that original variables are not in transformed dataframe
    assert set(vars).isdisjoint(set(X_tr.columns))
    # Check that remaining variables are in transformed dataframe
    remaining = [f for f in estimator.feature_names_in_ if f not in vars]
    assert all([f in X_tr.columns for f in remaining])

    # when drop_original is False
    estimator = clone(estimator)
    estimator.set_params(drop_original=False)
    X_tr = estimator.fit_transform(X, y)

    vars = estimator.variables_

    # Check that original variables are in transformed dataframe
    assert len([f in X_tr.columns for f in vars])
    # Check that remaining variables are in transformed dataframe
    remaining = [f for f in estimator.feature_names_in_ if f not in vars]
    assert all([f in X_tr.columns for f in remaining])


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


def check_raises_error_if_only_1_variable(estimator):
    """For feature selection transformers.

    Checks that the transformer has 2 or more
    variables to select from during the search procedure.
    """
    X, y = test_df()
    estimator = clone(estimator)
    sel = estimator.set_params(
        variables=["var_1"],
        confirm_variables=False,
    )

    msg = (
        "The selector needs at least 2 or more variables to select from. "
        "Got only 1 variable: ['var_1']."
    )
    with pytest.raises(ValueError) as record:
        sel.fit(X, y)

    assert str(record.value) == msg
