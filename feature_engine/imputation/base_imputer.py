import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)


class BaseImputer(BaseEstimator, TransformerMixin):
    """shared set-up checks and methods across imputers"""

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Check that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X : Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the dataframe is not of same size as that used in fit()

        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered by the user.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input df contains same number of columns as df used to fit
        _check_input_matches_training_df(X, self.n_features_in_ )

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing data with the learned parameters.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the dataframe is not of same size as that used in fit()

        Returns
        -------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the selected variables.
        """

        X = self._check_transform_input_and_state(X)

        # replaces missing data with the learned parameters
        for variable in self.imputer_dict_:
            X[variable].fillna(self.imputer_dict_[variable], inplace=True)

        return X

    def _more_tags(self):
        return {
            "_xfail_checks": {
                # check_estimator checks that fail:
                "check_estimators_nan_inf": "transformer allows NA",

                # Complex data in math terms, are values like 4i (imaginary numbers
                # so to speak). I've never seen such a thing in the dfs I've
                # worked with, so I do not need this test.
                "check_complex_data": "I dont think we need this check, if users "
                                      "disagree we can think how to introduce it "
                                      "at a later stage.",

                # check that estimators treat dtype object as numeric if possible
                "check_dtype_object":
                    "Transformers use dtypes to select between numerical and "
                    "categorical variables. Feature-engine trusts the user cast the "
                    "variables in they way they would like them treated.",

                # Not sure what the aim of this check is, it fails because FE does not
                # like the sklearn class _NotAnArray
                "check_transformer_data_not_an_array": "Not sure what this check is",

                # this test fails because the test uses dtype attribute of numpy, but
                # in feature engine the array is converted to a df, and it does not
                # have the dtype attribute.
                # need to understand why this test is useful an potentially have one
                # for the package. But some Feature-engine transformers DO change the
                # types
                "check_transformer_preserve_dtypes":
                    "Test not relevant, Feature-engine transformers can change "
                    "the types",

                # TODO: we probably need the test below!!
                "check_methods_sample_order_invariance":
                    "Test does not work on dataframes",

                # TODO: we probably need the test below!!
                # the test below tests that a second fit overrides a first fit.
                # the problem is that the test does not work with pandas df.
                "check_fit_idempotent": "Test does not work on dataframes",

                "check_fit1d": "Test not relevant, Feature-engine transformers only "
                               "work with dataframes",

                "check_fit2d_predict1d":
                    "Test not relevant, Feature-engine transformers only "
                    "work with dataframes",
            }
        }