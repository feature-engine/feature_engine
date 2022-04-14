from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import deprecated
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import (
    _drop_original_docstring,
    _missing_values_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.tags import _return_tags
from feature_engine.variable_manipulation import _find_or_check_numerical_variables


@deprecated(
    "CombineWithReferenceFeature() is deprecated in version 1.3 and will be removed in "
    "version 1.4. Use RelativeFeatures() instead."
)
@Substitution(
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class CombineWithReferenceFeature(BaseEstimator, TransformerMixin):
    """
    **DEPRECATED: CombineWithReferenceFeature() is deprecated in version 1.3 and will be
    removed in Version 1.4. Use RelativeFeatures() instead.**

    CombineWithReferenceFeature() applies basic mathematical operations between a group
    of variables and one or more reference features. It adds one or more additional
    features to the dataframe with the result of the operations.

    In other words, CombineWithReferenceFeature() sums, multiplies, subtracts or
    divides a group of features to / by a group of reference variables, and returns the
    result as new variables in the dataframe.

    The transformed dataframe will contain the additional features indicated in the
    new_variables_name list plus the original set of variables.

    More details in the :ref:`User Guide <combine_with_ref>`.

    Parameters
    ----------
    variables_to_combine: list
        The list of numerical variables to combine with the reference variables.

    reference_variables: list
        The list of numerical reference variables that will be added to, multiplied
        with, or subtracted from the `variables_to_combine`, or used as denominator for
        division.

    operations: list, default=['sub']
        The list of basic mathematical operations to be used in the transformation.

        If None, all of ['sub', 'div', 'add', 'mul'] will be performed. Alternatively,
        you can enter a list of operations to carry out. Each operation should
        be a string and must be one of the elements in `['sub', 'div', 'add', 'mul']`.

        Each operation will result in a new variable that will be added to the
        transformed dataset.

    new_variables_names: list, default=None
        Names of the new variables. If passing a list with the names for the new
        features (recommended), you must enter as many names as new features created
        by the transformer. The number of new features is the number of `operations`,
        times the number of `reference_variables`, times the number of
        `variables_to_combine`.

        If `new_variable_names` is None, the transformer will assign an arbitrary name
        to the features. The name will be var + operation + ref_var.

    {missing_values}

    {drop_original}

    Attributes
    ----------
    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    transform:
        Create new features.

    {fit_transform}

    Notes
    -----
    Although the transformer in essence allows us to combine any feature with any of
    the allowed mathematical operations, its use is intended mostly for the creation
    of new features based on some domain knowledge. Typical examples within the
    financial sector are:

    - Ratio between income and debt to create the debt_to_income_ratio.
    - Subtraction of rent from income to obtain the disposable_income.
    """

    def __init__(
        self,
        variables_to_combine: List[Union[str, int]],
        reference_variables: List[Union[str, int]],
        operations: List[str] = ["sub"],
        new_variables_names: Optional[List[str]] = None,
        missing_values: str = "ignore",
        drop_original: bool = False,
    ) -> None:

        # check input types
        if not isinstance(reference_variables, list) or not all(
            isinstance(var, (int, str)) for var in reference_variables
        ):
            raise ValueError(
                "reference_variables takes a list of strings or integers "
                "corresponding to the names of the variables to be used as  "
                "reference to combine with the binary operations."
            )

        if not isinstance(variables_to_combine, list) or not all(
            isinstance(var, (int, str)) for var in variables_to_combine
        ):
            raise ValueError(
                "variables_to_combine takes a list of strings or integers "
                "corresponding to the names of the variables to combine "
                "with the binary operations."
            )

        if new_variables_names:
            if not isinstance(new_variables_names, list) or not all(
                isinstance(var, str) for var in new_variables_names
            ):
                raise ValueError(
                    "new_variable_names should be None or a list with the "
                    "names to be assigned to the new variables created by"
                    "the mathematical combinations."
                )

        if operations:
            if not isinstance(operations, list):
                raise ValueError("operations parameter must be a list or None")

            if any(
                operation not in ["sub", "div", "add", "mul"]
                for operation in operations
            ):
                raise ValueError(
                    "At least one of the entered operations is not supported. "
                    "Choose one or more of ['sub', 'div','add','mul']"
                )

        if new_variables_names:
            if len(new_variables_names) != (
                len(reference_variables) * len(variables_to_combine) * len(operations)
            ):
                raise ValueError(
                    "Number of items in new_variables_names must be equal to number of "
                    "items in reference_variables * items in variables to "
                    "combine * binary operations. In other words, "
                    "the transformer needs as many new variable names as reference "
                    "variables and binary operations to perform over the variables to "
                    "combine."
                )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        if not isinstance(drop_original, bool):
            raise TypeError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )

        self.reference_variables = reference_variables
        self.variables_to_combine = variables_to_combine
        self.new_variables_names = new_variables_names
        self.operations = operations
        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, or np.array. Default=None.
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = check_X(X)

        # check variables to combine are numerical
        self.variables_to_combine = _find_or_check_numerical_variables(
            X, self.variables_to_combine
        )

        # check reference_variables are numerical
        self.reference_variables = _find_or_check_numerical_variables(
            X, self.reference_variables
        )

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.reference_variables)
            _check_contains_na(X, self.variables_to_combine)

            _check_contains_inf(X, self.reference_variables)
            _check_contains_inf(X, self.variables_to_combine)

        # cannot divide by 0, as will result in error
        if "div" in self.operations:
            if X[self.reference_variables].isin([0]).any().any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer with div."
                )

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Combine the variables with the mathematical operations.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features + n_operations]
            The dataframe with the new variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.reference_variables)
            _check_contains_na(X, self.variables_to_combine)

            _check_contains_inf(X, self.reference_variables)
            _check_contains_inf(X, self.variables_to_combine)

        # cannot divide by 0, as will result in error
        if "div" in self.operations:
            if X[self.reference_variables].isin([0]).any().any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )

        # Add new features and values into de data frame.
        if "sub" in self.operations:
            for reference in self.reference_variables:
                varname = [
                    str(var) + "_sub_" + str(reference)
                    for var in self.variables_to_combine
                ]
                X[varname] = X[self.variables_to_combine].sub(X[reference], axis=0)
        if "div" in self.operations:
            for reference in self.reference_variables:
                varname = [
                    str(var) + "_div_" + str(reference)
                    for var in self.variables_to_combine
                ]
                X[varname] = X[self.variables_to_combine].div(X[reference], axis=0)
        if "add" in self.operations:
            for reference in self.reference_variables:
                varname = [
                    str(var) + "_add_" + str(reference)
                    for var in self.variables_to_combine
                ]
                X[varname] = X[self.variables_to_combine].add(X[reference], axis=0)
        if "mul" in self.operations:
            for reference in self.reference_variables:
                varname = [
                    str(var) + "_mul_" + str(reference)
                    for var in self.variables_to_combine
                ]
                X[varname] = X[self.variables_to_combine].mul(X[reference], axis=0)

        # replace created variable names with user ones.
        if self.new_variables_names:
            X.columns = self.feature_names_in_ + self.new_variables_names

        if self.drop_original:
            X.drop(
                columns=set(self.variables_to_combine + self.reference_variables),
                inplace=True,
            )

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"

        msg = "this transformer works with datasets that contain at least 2 variables. \
        Otherwise, there is nothing to combine"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg
        return tags_dict
