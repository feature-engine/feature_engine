from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.variable_manipulation import _find_or_check_numerical_variables


class CombineWithReferenceFeature(BaseEstimator, TransformerMixin):
    """
    CombineWithReferenceFeature() applies basic mathematical operations between one or
    more reference variables and a group of variables, returning one or more additional
    features as a result. That is, it sums, multiplies, substracts or divides a group of
    features to or by a group of reference variables and returns the result into new
    variables.

    For example, if we have the variables **number_payments_first_quarter**,
    **number_payments_second_quarter**, **number_payments_third_quarter**,
    **number_payments_fourth_quarter**, and **total_payments** we can use
    CombineWithReferenceFeature() to determine the percentage of total payments per
    month as follows:

    .. code-block:: python

        transformer = CombineWithReferenceFeature(
            variables_to_combine=[
                'number_payments_first_quarter',
                'number_payments_second_quarter',
                'number_payments_third_quarter',
                'number_payments_fourth_quarter',
            ],

            reference_variables=['total_payments'],

            operations=['div'],

            new_variables_name=[
                'perc_payments_first_quarter',
                'perc_payments_second_quarter',
                'perc_payments_third_quarter',
                'perc_payments_fourth_quarter',
            ]
        )

        Xt = transformer.fit_transform(X)

    The transformed X, Xt, will contain the additional features indicated in the
    new_variables_name list plus the original set of variables.

    Parameters
    ----------

    variables_to_combine : list
        The list of numerical variables to be combined with the reference
        variables.

    reference_variables : list
        The list of numerical reference variables that will be added, multiplied,
        or substracted from the variables_to_combine, or used as denominator for
        division.

    operations : list, default=['sub']
        The list of basic mathematical operations to be used in transformation.

        If none, all of ['sub', 'div','add','mul'] will be performed
        over the variables. Alternatively, the user can enter the list of
        operations to carry out.

        Each operation should be a string and must be one of the elements
        from the list: ['sub', 'div','add','mul']

        Each operation will result in a new variable that will be added to the
        transformed dataset.

    new_variables_names : list, default=None
        Names of the newly created variables. The user can enter a list with the
        names for the newly created features (recommended). The user must enter
        as many names as new features created by the transformer. The number of new
        features is the number of operations times the number of reference variables
        times the number of variables to combine.

        Thus, if you want to perform 2 operations, sub and div, combining 4 variables
        with 2 reference variables, you should enter 2 X 4 X 2 new variable names.

        The name of the variables indicated by the user should coincide with the order
        in which the  operations are performed by the transformer. The transformer will
        first carry out 'sub', then 'div', then 'add' and finally 'mul'.

        If new_variable_names=None, the transformer will assign an arbitrary name
        to the newly created features.

    missing_values : string, default='ignore'
        Indicates if missing values should be ignored or raised. If
        missing_values='ignore', the transformer will ignore missing data when
        transforming the data. If missing_values='raise' the transformer will return
        an error if the training or the datasets to transform contain missing values.

    Methods
    -------

    fit :
        This transformer does not learn parameters.
    transform :
        Combine the variables with the mathematical operations.
    fit_transform :
        Fit to the data, then transform it.

    Notes
    -----
    Although the transformer in essence allows us to combine any feature with any of
    the allowed mathematical operations, its used is intended mostly for the creation
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

        self.reference_variables = reference_variables
        self.variables_to_combine = variables_to_combine
        self.new_variables_names = new_variables_names
        self.operations = operations
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.
        Performs dataframe checks.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
        The training input samples.
        Can be the entire dataframe, not just the variables to transform.

        y : pandas Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
           - If the input is not a Pandas DataFrame
           - If any user provided variables are not numerical
        ValueError
           If any of the reference variables contain null values and the
           mathematical operation is 'div'.

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

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

        # cannot divide by 0, as will result in error
        if "div" in self.operations:
            if X[self.reference_variables].isin([0]).any().any():
                raise ValueError(
                    "Some of the reference variables contain 0 values. Check and "
                    "remove those before using this transformer with div."
                )

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Combine the variables with the mathematical operations.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
        The data to transform.

        Returns
        -------

        X : Pandas dataframe, shape = [n_samples, n_features + n_operations]
        The dataframe with the operations results added as columns.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.reference_variables)
            _check_contains_na(X, self.variables_to_combine)

        # cannot divide by 0, as will result in error
        if "div" in self.operations:
            if X[self.reference_variables].isin([0]).any().any():
                raise ValueError(
                    "Some of the reference variables contain 0 values. Check and "
                    "remove those before using this transformer."
                )

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.input_shape_[1])

        original_col_names = [var for var in X.columns]
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
            X.columns = original_col_names + self.new_variables_names

        return X
