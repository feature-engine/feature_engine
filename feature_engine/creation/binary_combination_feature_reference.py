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


class BinaryCombinatioByFeatureReference(BaseEstimator, TransformerMixin):
    """
    BinaryCombinatioByFeatureReference() applies binary operations across targert and reference features,
    returning 1 or more additional features as a result.

    For example ...

    Parameters
    ----------

    Reference_variables: list
        The list of numerical reference variables to operate.

    variables_to_combine: list
        The list of target numerical variables to be combined.

    binary_operations: list, default=None
        The list of basic binary operations to be used in transformation.

        If none, all of ['sub', 'div'] will be performed
        over the variables. Alternatively, user can enter the list of operations to
        carry out.

        Each operation should be a string and must be one of the elements
        from the list: ['sub', 'div']

        Each operation will result in a new variable that will be added to the
        transformed dataset.

    new_variables_names: list, default=None
        Names of the newly created variables. The user can enter a name or a list
        of names for the newly created features (recommended). User must enter
        one name for each binary transformation indicated in the math_operations
        attribute. That is, if you want to perform mean and sum of features, you
        should enter 2 new variable names. If you perform only mean of features,
        enter 1 variable name. Alternatively, if you chose to perform all
        binary transformations, enter 6 new variable names.

        The name of the variables indicated by the user should coincide with the order
        in which the binary operations are initialised in the transformer.
        That is, if you set math_operations = ['mean', 'prod'], the first new variable
        name will be assigned to the mean of the variables and the second variable name
        to the product of the variables.

        If new_variable_names=None, the transformer will assign an arbitrary name
        to the newly created features starting by the name of the binary
        operation, followed by the variables combined separated by -.

    """

    def __init__(
        self,
        variables_to_combine: List[Union[str, int]],
        reference_variables: List[Union[str, int]],
        binary_operations: [List[str]] = "sub",
        new_variables_names: Optional[List[str]] = None,
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

        if binary_operations:
            if not isinstance(binary_operations, list):
                raise ValueError("binary_operations parameter must be a list or None")

            if any(
                operation not in ["sub", "div"]
                for operation in binary_operations
            ):
                raise ValueError(
                    "At least one of the entered binary_operations is not supported. "
                    "Choose one or more of ['sub', 'div']"
                )

        # check input logic
        if len(reference_variables) <= 0:
            raise KeyError(
              "reference_variables requires one or more features to make proper "
              "transformations."
          )

        if new_variables_names:
            if len(new_variables_names) != len(reference_variables):  # type: ignore
                raise ValueError(
                    "Number of items in new_variables_names must be equal to number of "
                    "items in Reference_variables."
                    "In other words, the transformer needs as many new variable names"
                    "as reference variables to perform over the variables to "
                    "combine."
                )

        self.reference_variables = reference_variables
        self.variables_to_combine = variables_to_combine
        self.new_variables_names = new_variables_names
        self.binary_operations = binary_operations

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Performs dataframe checks.

        Args:
            X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

            y: It is not needed in this transformer.
            Defaults to None. Alternatively takes Pandas Series

        Returns:
            self
        """
        # check input dataframe
        X = _is_dataframe(X)

        # check variables to combine are numerical
        self.variables_to_combine = _find_or_check_numerical_variables(
            X, self.variables_to_combine
        )

        # check if dataset contains na
        _check_contains_na(X, self.reference_variables)

        # cannot divide by 0, as will result in error
        if 'sub' in self.binary_operations:
            if X[self.reference_variables].isin([0]).any().any():
                raise ValueError(
                    "Some of the refence variables contain 0 values. Check and "
                    "remove those before using this transformer."
            )

        # dictionary of new_variable_name to operation pairs
        if self.new_variables_names:
            self.combination_dict_ = dict(
                zip(self.new_variables_names, self.binary_operations)
            )
        else:

            # NG - Not enought to support this class new variable names
            if all(isinstance(var, str) for var in self.variables_to_combine):
                vars_ls = self.variables_to_combine
            else:
                vars_ls = [str(var) for var in self.variables_to_combine]

            self.combination_dict_ = {
                f"{operation}({'-'.join(vars_ls)})": operation  # type: ignore
                for operation in self.binary_operations
            }

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms source dataset.

        Adds a column for each operation with the calculation based on the variables
        and operations indicated when setting up the transformer.

        Args:
            X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns:
            Pandas dataframe, shape = [n_samples, n_features + n_operations]
            The dataframe with the operations results added as columns.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables_to_combine)

        # cannot divide by 0, as will result in error
        if 'sub' in self.binary_operations:
            if X[self.reference_variables].isin([0]).any().any():
                raise ValueError(
                    "Some of the refence variables contain 0 values. Check and "
                    "remove those before using this transformer."
            )

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.input_shape_[1])
 
        # makes de magic 
        # NG - No way this will works.
        for variables_to_combine in self.variables_to_combine:
            for reference_variables in self.reference_variables:
                for new_variable_name, operation in self.combination_dict_.items():
                    X[new_variable_name] = X[variables_to_combine,reference_variables].agg(operation, axis=1)

        return X
