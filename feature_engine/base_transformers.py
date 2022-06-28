""" The base transformer provides functionality that is shared by most transformer
classes. Provides the base functionality within the fit() and transform() methods
shared by most transformers, like checking that input is a df, the size, NA, etc.
"""
from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.get_feature_names_out import _get_feature_names_out
from feature_engine._docstrings.methods import _get_feature_names_out_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.tags import _return_tags
from feature_engine.variable_manipulation import _find_or_check_numerical_variables


class BaseNumericalTransformer(BaseEstimator, TransformerMixin):
    """Shared set-up procedures across numerical transformers, i.e.,
    variable transformers, discretisers, math combination.
    """

    def _fit_from_dict(self, X: pd.DataFrame, user_dict_: Dict) -> pd.DataFrame:
        """
        Checks that input is a dataframe, checks that variables in the dictionary
        entered by the user are of type numerical.

        Parameters
        ----------
        X : Pandas DataFrame

        user_dict_ : Dictionary. Default = None
            Any dictionary allowed by the transformer and entered by user.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame or a numpy array
            If any of the variables in the dictionary are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
            If the variable(s) contain null values

        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        """
        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        variables = [x for x in user_dict_.keys()]
        self.variables_ = _find_or_check_numerical_variables(X, variables)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return X

    def _fit_from_varlist(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that input is a dataframe, finds numerical variables, or alternatively
        checks that variables entered by the user are of type numerical.

        Parameters
        ----------
        X : Pandas DataFrame

        y : Pandas Series, np.array. Default = None
            Parameter is necessary for compatibility with sklearn Pipeline.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame or a numpy array
            If any of the user provided variables are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
            If the variable(s) contain null values

        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        """

        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit() method. Checks absence of NA and Inf.

        Parameters
        ----------
        X : Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X : Pandas DataFrame.
            The same dataframe entered by the user.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        return X

    @Substitution(get_feature_names_out=_get_feature_names_out_docstring)
    def get_feature_names_out(
        self, input_features: Optional[List[Union[str, int]]] = None
    ) -> List[Union[str, int]]:
        """{get_feature_names_out}"""

        check_is_fitted(self)

        feature_names = _get_feature_names_out(
            features_in=self.feature_names_in_,
            transformed_features=self.variables_,
            input_features=input_features,
        )

        return feature_names

    # for the check_estimator tests
    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        return tags_dict
