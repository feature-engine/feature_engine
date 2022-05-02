from typing import Any, List, Optional, Union

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine.creation.base_creation import BaseCreation
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


@Substitution(
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    transform=BaseCreation._transform_docstring,
    fit_transform=_fit_transform_docstring,
)
class MathFeatures(BaseCreation):
    """
    MathFeatures(() applies functions across multiple features returning one or more
    additional features as a result. It uses `pandas.agg()` to create the features,
    setting `axis=1`.

    For supported aggregation functions, see `pandas documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html>`_.

    Note that if some of the variables have missing data and `missing_values='ignore'`,
    the value will be ignored in the computation. To be clear, if variables A, B and C,
    have values 10, 20 and NA, and we perform the sum, the result will be A + B = 30.

    More details in the :ref:`User Guide <math_features>`.

    Parameters
    ----------
    variables: list
        The list of input variables. Variables must be numerical and there must be at
        least 2 different variables in the list.

    func: function, string, list
        Functions to use for aggregating the data. Same functionality as parameter
        `func` in `pandas.agg()`. If a function, it must either work when passed a
        DataFrame or when passed to DataFrame.apply. Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. [np.sum, 'mean']

        Each function will result in a new variable that will be added to the
        transformed dataset.

    new_variables_names: list, default=None
        Names of the new variables. If passing a list with names (recommended), enter
        one name per function. If None, the transformer will assign arbitrary names,
        starting with the function and followed by the variables separated by _.

    {missing_values}

    {drop_original}

    Attributes
    ----------
    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    Notes
    -----
    Although the transformer allows us to combine any features with any functions, we
    recommend using it to create features based on domain knowledge. Typical examples
    in finance are:

    - Sum debt across financial products, i.e., credit cards, to obtain the total debt.
    - Take the average payments to various financial products.
    - Find the minimum payment done at any one month.

    In insurance, we can sum the damage to various parts of a car to obtain the
    total damage.
    """

    def __init__(
        self,
        variables: List[Union[str, int]],
        func: Any,
        new_variables_names: Optional[List[str]] = None,
        missing_values: str = "raise",
        drop_original: bool = False,
    ) -> None:

        if (
            not isinstance(variables, list)
            or not all(isinstance(var, (int, str)) for var in variables)
            or len(variables) < 2
            or len(set(variables)) != len(variables)
        ):
            raise ValueError(
                "variables must be a list of strings or integers with at least 2 "
                f"different variables. Got {variables} instead."
            )

        if isinstance(func, dict):
            raise NotImplementedError(
                "func does not work with dictionaries in this transformer."
            )

        if new_variables_names is not None:
            if (
                not isinstance(new_variables_names, list)
                or not all(isinstance(var, str) for var in new_variables_names)
                or len(set(new_variables_names)) != len(new_variables_names)
            ):
                raise ValueError(
                    "new_variable_names should be None or a list of unique strings. "
                    f"Got {new_variables_names} instead."
                )

        if new_variables_names is not None:
            if isinstance(func, list):
                if len(new_variables_names) != len(func):
                    raise ValueError(
                        "The number of new feature names must coincide with the number "
                        "of functions."
                    )
            else:
                if len(new_variables_names) != 1:
                    raise ValueError(
                        "The number of new feature names must coincide with the number "
                        "of functions."
                    )

        super().__init__(missing_values, drop_original)

        self.variables = variables
        self.func = func
        self.new_variables_names = new_variables_names

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to aggregate.

        y: pandas Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.
        """
        # Common checks and attributes
        super().fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create and add new variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features + n_operations]
            The input dataframe plus the new variables.
        """
        X = super().transform(X)

        new_variable_names = self.get_feature_names_out(input_features=True)

        if len(new_variable_names) == 1:
            X[new_variable_names[0]] = X[self.variables].agg(self.func, axis=1)
        else:
            X[new_variable_names] = X[self.variables].agg(self.func, axis=1)

        if self.drop_original:
            X.drop(columns=self.variables, inplace=True)

        return X

    def get_feature_names_out(self, input_features: Optional[bool] = None) -> List:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: bool, default=None
            If `input_features` is `None`, then the names of all the variables in the
            transformed dataset (original + new variables) is returned. Alternatively,
            if `input_features` is True, only the names for the new features will be
            returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        if input_features is not None and not isinstance(input_features, bool):
            raise ValueError(
                "input_features takes None or a boolean, True or False. "
                f"Got {input_features} instead."
            )

        # create name of the new variables
        if self.new_variables_names is not None:
            feature_names = self.new_variables_names

        else:
            varlist = [str(var) for var in self.variables]

            if isinstance(self.func, list):
                functions = [
                    fun if type(fun) == str else fun.__name__ for fun in self.func
                ]
                feature_names = [
                    f"{function}_{'_'.join(varlist)}" for function in functions
                ]
            else:
                feature_names = [f"{self.func}_{'_'.join(varlist)}"]

        # organise return of method
        if input_features is None or input_features is False:
            if self.drop_original is True:
                # Remove names of variables to drop.
                original = [
                    f for f in self.feature_names_in_ if f not in self.variables
                ]
                feature_names = original + feature_names
            else:
                feature_names = self.feature_names_in_ + feature_names

        return feature_names
