from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _check_X_matches_training_df, check_X
from feature_engine.docstrings import Substitution, _input_features_docstring
from feature_engine.tags import _return_tags


class BaseImputer(BaseEstimator, TransformerMixin):
    """shared set-up checks and methods across imputers"""

    _variables_numerical_docstring = """variables: list, default=None
        The list of variables to impute. If None, the imputer will select
        all numerical variables.
        """.rstrip()

    _imputer_dict_docstring = """imputer_dict_:
        Dictionary with the values to replace missing data in each variable.
        """.rstrip()

    _transform_docstring = """transform:
        Impute missing data.
        """.rstrip()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Common checks before transforming data:

        - Check transformer was fit
        - Check that the input is a dataframe
        - Check that input has same size than the train set used in fit()
        - Re-orders dataframe features if necessary

        Parameters
        ----------
        X: Pandas DataFrame

        Returns
        -------
        X: Pandas DataFrame
            The same dataframe entered by the user.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check that input df contains same number of columns as df used to fit
        _check_X_matches_training_df(X, self.n_features_in_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing data with the learned parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the selected variables.
        """

        X = self._transform(X)

        # Replace missing data with learned parameters
        X.fillna(value=self.imputer_dict_, inplace=True)

        return X

    def _get_feature_names_in(self, X):
        """Get the names and number of features in the train set (the dataframe
        used during fit)."""

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self

    @Substitution(
        input_features=_input_features_docstring,
    )
    def get_feature_names_out(
            self, input_features: Optional[Union[List, str]] = None
    ) -> List:
        """Get output feature names for transformation.

        Parameters
        ----------
        {input_features}

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        if input_features is None:
            # return all feature names
            feature_names = [
                f for f in self.feature_names_in_ if f not in self.features_to_drop_
            ]

        else:
            # Return features requested by user.
            if not isinstance(input_features, list):
                raise ValueError(
                    f"input_features must be a list. Got {input_features} instead."
                )
            if any([f for f in input_features if f not in self.variables_]):
                raise ValueError(
                    "Some features in input_features were not used to extract new "
                    "variables. Pass either None, or a list with the features that "
                    "were used to create date and time features."
                )
            feature_names = input_features

        return feature_names

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "numerical"
        return tags_dict
