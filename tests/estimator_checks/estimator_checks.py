import pandas as pd
import pytest
from sklearn.base import clone

from tests.estimator_checks.dataframe_for_checks import test_df
from tests.estimator_checks.fit_functionality_checks import (
    check_error_if_y_not_passed,
    check_feature_names_in,
)
from tests.estimator_checks.get_feature_names_out_checks import (
    check_get_feature_names_out,
)
from tests.estimator_checks.init_params_allowed_values_checks import (
    check_error_param_missing_values,
)
from tests.estimator_checks.init_params_triggered_functionality_checks import (
    check_drop_original_variables,
    check_takes_cv_constructor,
)
from tests.estimator_checks.non_fitted_error_checks import check_raises_non_fitted_error
from tests.estimator_checks.variable_selection_checks import (
    check_all_types_variables_assignment,
    check_categorical_variables_assignment,
    check_datetime_variables_assignment,
    check_numerical_variables_assignment,
)


def check_feature_engine_estimator(estimator, needs_group: bool = False):
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
            check_numerical_variables_assignment(estimator, needs_group=needs_group)
        elif tags["variables"] == "categorical":
            check_categorical_variables_assignment(estimator, needs_group=needs_group)
        elif tags["variables"] == "all":
            check_all_types_variables_assignment(estimator, needs_group=needs_group)
        elif tags["variables"] == "datetime":
            check_datetime_variables_assignment(estimator)
        else:
            pass

    # Tests based on transformer's init parameters
    if hasattr(estimator, "cv"):
        check_takes_cv_constructor(estimator)

    if hasattr(estimator, "missing_values"):
        check_error_param_missing_values(estimator)

    if hasattr(estimator, "drop_original"):
        check_drop_original_variables(estimator)

    return None


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
