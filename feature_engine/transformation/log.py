# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import _check_input_parameter_variables


class LogTransformer(BaseNumericalTransformer):
    """
    The LogTransformer() applies the natural logarithm or the base 10 logarithm to
    numerical variables. The natural logarithm is the logarithm in base e.

    The LogTransformer() only works with positive values. If the variable
    contains a zero or a negative value the transformer will return an error.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the transformer
        will find and select all numerical variables.

    base: string, default='e'
        Indicates if the natural or base 10 logarithm should be applied. Can take
        values 'e' or '10'.

    Attributes
    ----------
    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        This transformer does not learn parameters.
    transform:
        Transform the variables using the logarithm.
    fit_transform:
        Fit to data, then transform it.
    inverse_transform:
        Convert the data back to the original representation.
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        base: str = "e",
    ) -> None:

        if base not in ["e", "10"]:
            raise ValueError("base can take only '10' or 'e' as values")

        self.variables = _check_input_parameter_variables(variables)
        self.base = base

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Selects the numerical variables and determines whether the logarithm
        can be applied on the selected variables, i.e., it checks that the variables
        are positive.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            - If there are no numerical variables in the df or the df is empty
            - If the variable(s) contain null values
            - If some variables contain zero or negative values

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        # check contains zero or negative values
        if (X[self.variables_] <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values, can't apply log"
            )

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the variables using log transformation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()
            - If some variables contains zero or negative values

        Returns
        -------
        X: pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # check contains zero or negative values
        if (X[self.variables_] <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values, can't apply log"
            )

        # transform
        if self.base == "e":
            X.loc[:, self.variables_] = np.log(X.loc[:, self.variables_])
        elif self.base == "10":
            X.loc[:, self.variables_] = np.log10(X.loc[:, self.variables_])

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()
            - If some variables contains zero or negative values

        Returns
        -------
        X: pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # inverse_transform
        if self.base == "e":
            X.loc[:, self.variables_] = np.exp(X.loc[:, self.variables_])
        elif self.base == "10":
            X.loc[:, self.variables_] = np.array(10 ** X.loc[:, self.variables_])

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # =======  this tests fail because the transformers throw an error
        # when the values are 0. Nothing to do with the test itself but
        # mostly with the data created and used in the test
        msg = (
            "transformers raise errors when data contains zeroes, thus this check fails"
        )
        tags_dict["_xfail_checks"]["check_estimators_dtypes"] = msg
        tags_dict["_xfail_checks"]["check_estimators_fit_returns_self"] = msg
        tags_dict["_xfail_checks"]["check_pipeline_consistency"] = msg
        tags_dict["_xfail_checks"]["check_estimators_overwrite_params"] = msg
        tags_dict["_xfail_checks"]["check_estimators_pickle"] = msg
        tags_dict["_xfail_checks"]["check_transformer_general"] = msg

        return tags_dict


class LogCpTransformer(BaseNumericalTransformer):
    """
    The LogCpTransformer() applies the transformation log(x + C), where C is a positive
    constant, to the input variable. It applies the natural logarithm or the base 10
    logarithm, where the natural logarithm is logarithm in base e.

    The LogCpTransformer() only works with numerical non-negative values. If the
    variable contains a zero or a negative value after adding a constant C, the
    transformer will return an error.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the transformer
        will find and select all numerical variables.

    base: string, default='e'
        Indicates if the natural or base 10 logarithm should be applied. Can take
        values 'e' or '10'.

    C: "auto" or int, default="auto"
        The constant C to apply the transfomation log(x + C).

        - If int, then log(x + C)
        - If "auto", then C = abs(min(x)) + 1

    Attributes
    ----------
    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        This transformer does not learn parameters.
    transform:
        Transforms the variables using log transformation.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        base: str = "e",
        C: Union[int, str] = "auto",
    ) -> None:

        if base not in ["e", "10"]:
            raise ValueError("base can take only '10' or 'e' as values")

        if not isinstance(C, int) and not C == "auto":
            raise ValueError("C can take only 'auto' or int")

        self.variables = _check_input_parameter_variables(variables)
        self.base = base
        self.C = C

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Select the numerical variables and determines whether the logarithm
        can be applied on the selected variables (it checks if the variables
        are all positive).

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            - If there are no numerical variables in the df or the df is empty
            - If the variable(s) contain null values
            - If some variables contain zero or negative values after suming constant C

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        # calculate C to add to each variable
        if self.C == "auto":
            self.C = X.min(axis=0).abs() + 1

        # check contains zero or negative values
        if (X[self.variables_] + self.C <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values after suming"
                + "constant C, can't apply log"
            )

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the variables using log transformation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()
            - If some variables contains zero or negative values after suming constant C

        Returns
        -------
        X: pandas dataframe
            The dataframe with the transformed variables.
        """
        # TO DO: check X is pandas DataFrame

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # check contains zero or negative values
        if (X[self.variables_] + self.C <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values after suming"
                + "constant C, can't apply log"
            )

        # transform
        if self.base == "e":
            X.loc[:, self.variables_] = np.log(X.loc[:, self.variables_] + self.C)
        elif self.base == "10":
            X.loc[:, self.variables_] = np.log10(X.loc[:, self.variables_] + self.C)

        return X
