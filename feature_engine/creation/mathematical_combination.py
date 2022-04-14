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
    "MathematicalCombination() is deprecated in version 1.3 and will be removed in "
    "version 1.4. Use MathFeatures() instead."
)
@Substitution(
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class MathematicalCombination(BaseEstimator, TransformerMixin):
    """
    **DEPRECATED: MathematicalCombination() is deprecated in version 1.3 and will be
    removed in version 1.4. Use MathFeatures() instead.**

    MathematicalCombination() applies basic mathematical operations to multiple
    features, returning one or more additional features as a result. That is, it sums,
    multiplies, takes the average, maximum, minimum or standard deviation of a group
    of variables, and returns the result into new variables.

    Note that if some of the variables to combine have missing data and you set
    `missing_values='ignore'`, the value will be ignored in the computation. To be
    clear, if variables A, B and C, have values 10, 20 and NA, and we perform the sum,
    the result will be A + B = 30.

    More details in the :ref:`User Guide <math_combination>`.

    Parameters
    ----------
    variables_to_combine: list
        The list of numerical variables to combine.

    math_operations: list, default=None
        The list of basic math operations to be used to create the new features.

        If None, all of ['sum', 'prod', 'mean', 'std', 'max', 'min'] will be performed.
        Alternatively, you can enter the list of operations to carry out. Each operation
        should be a string and must be one of the elements in
        `['sum', 'prod', 'mean', 'std', 'max', 'min']`.

        Each operation will result in a new variable that will be added to the
        transformed dataset.

    new_variables_names: list, default=None
        Names of the new variables. If passing a list with the names for the new
        features (recommended), you must enter one name for each mathematical
        transformation indicated in the `math_operations` parameter. The name of the
        new variables should coincide with the order in which the mathematical
        operations are initialised in the transformer.

        If `new_variable_names = None`, the transformer will assign an arbitrary name
        to the newly created features starting by the name of the mathematical
        operation, followed by the variables combined separated by -.

    {missing_values}

    {drop_original}

    Attributes
    ----------
    combination_dict_:
        Dictionary containing the mathematical operation to new variable name pairs.

    math_operations_:
        List with the mathematical operations to be applied to the
        `variables_to_combine`.

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
    the allowed mathematical operations, its used is intended mostly for the creation
    of new features based on some domain knowledge. Typical examples within the
    financial sector are:

    - Sum debt across financial products, i.e., credit cards, to obtain the total debt.
    - Take the average payments to various financial products per month.
    - Find the Minimum payment done at any one month.

    In insurance, we can sum the damage to various parts of a car to obtain the
    total damage.
    """

    def __init__(
        self,
        variables_to_combine: List[Union[str, int]],
        math_operations: Optional[List[str]] = None,
        new_variables_names: Optional[List[str]] = None,
        missing_values: str = "raise",
        drop_original: bool = False,
    ) -> None:

        # check input types
        if not isinstance(variables_to_combine, list) or not all(
            isinstance(var, (int, str)) for var in variables_to_combine
        ):
            raise ValueError(
                "variables_to_combine takes a list of strings or integers "
                "corresponding to the names of the variables to combine "
                "with the mathematical operations."
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

        if math_operations:
            if not isinstance(math_operations, list):
                raise ValueError("math_operations parameter must be a list or None")

            if any(
                operation not in ["sum", "prod", "mean", "std", "max", "min"]
                for operation in math_operations
            ):
                raise ValueError(
                    "At least one of the entered math_operations is not supported. "
                    "Choose one or more of ['sum', 'prod', 'mean', 'std', 'max', 'min']"
                )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        # check input logic
        if len(variables_to_combine) <= 1:
            raise KeyError(
                "MathematicalCombination requires two or more features to make proper "
                "transformations."
            )

        if new_variables_names:
            if len(new_variables_names) != len(math_operations):  # type: ignore
                raise ValueError(
                    "Number of items in new_variables_names must be equal to number of "
                    "items in math_operations."
                    "In other words, the transformer needs as many new variable names"
                    "as mathematical operations to perform over the variables to "
                    "combine."
                )

        if not isinstance(drop_original, bool):
            raise TypeError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )

        self.variables_to_combine = variables_to_combine
        self.new_variables_names = new_variables_names
        self.math_operations = math_operations
        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Perform dataframe checks. Creates dictionary of operation to new feature
        name pairs.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = check_X(X)

        # check variables to combine are numerical
        self.variables_to_combine = _find_or_check_numerical_variables(
            X, self.variables_to_combine
        )

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_to_combine)
            _check_contains_inf(X, self.variables_to_combine)

        if self.math_operations is None:
            self.math_operations_ = ["sum", "prod", "mean", "std", "max", "min"]
        else:
            self.math_operations_ = self.math_operations

        # dictionary of new_variable_name to operation pairs
        if self.new_variables_names:
            self.combination_dict_ = dict(
                zip(self.new_variables_names, self.math_operations_)
            )
        else:
            if all(isinstance(var, str) for var in self.variables_to_combine):
                vars_ls = self.variables_to_combine
            else:
                vars_ls = [str(var) for var in self.variables_to_combine]

            self.combination_dict_ = {
                f"{operation}({'-'.join(vars_ls)})": operation  # type: ignore
                for operation in self.math_operations_
            }

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
            The dataframe with the original variables plus the new variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_to_combine)
            _check_contains_inf(X, self.variables_to_combine)

        # combine mathematically
        for new_variable_name, operation in self.combination_dict_.items():
            X[new_variable_name] = X[self.variables_to_combine].agg(operation, axis=1)

        if self.drop_original:
            X.drop(columns=self.variables_to_combine, inplace=True)

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"][
            "check_fit2d_1feature"
        ] = "this transformer works with datasets that contain at least 2 variables. \
        Otherwise, there is nothing to combine"
        return tags_dict
