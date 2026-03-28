""" The base transformer provides functionality that is shared by most transformer
classes. Provides the base functionality within the fit() and transform() methods
shared by most transformers, like checking that input is a df, the size, NA, etc.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


class BaseNumericalTransformer(
    TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin
):
    """Shared set-up procedures across numerical transformers, i.e.,
    variable transformers, discretisers, math combination.
    """

    def _fit_setup(self, X: pd.DataFrame):
        """
        Check dataframe, find numerical variables, check for NA and Inf.
        Returns the checked dataframe and the correctly identified numerical variables.
        """
        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        if self.variables is None:
            variables_ = find_numerical_variables(X)
        else:
            variables_ = check_numerical_variables(X, self.variables)

        # check if dataset contains na or inf
        _check_contains_na(X, variables_)
        _check_contains_inf(X, variables_)

        return X, variables_

    def _get_feature_names_in(self, X):
        """Get the names and number of features in the train set (the dataframe
        used during fit)."""

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
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

    # for the check_estimator tests
    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
