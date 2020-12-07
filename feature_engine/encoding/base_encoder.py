from typing import List, Union
import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_contains_na,
    _check_input_matches_training_df,
)
from feature_engine.variable_manipulation import _find_or_check_categorical_variables


class BaseCategoricalTransformer(BaseEstimator, TransformerMixin):
    """shared set-up checks and methods across categorical transformers"""

    def _check_fit_input_and_variables(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that input is a dataframe, finds categorical variables, or alternatively
        checks that the variables entered by the user are of type object (categorical).
        Checks absence of NA.

        Parameters
        ----------
        X : Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame.
            If any user provided variable is not categorical
        ValueError
            If there are no categorical variables in the df or the df is empty
            If the variable(s) contain null values

        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        variables : list
            list of categorical variables
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check variables entered by user are object
        self.variables: List[Union[str, int]] = _find_or_check_categorical_variables(
            X, self.variables
        )

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        return X

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X : Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the variable(s) contain null values.
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

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # Check input data contains same number of columns as df used to fit
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X

    def _check_encoding_dictionary(self):
        """After fit(), the encoders should return a dictionary with the original values
        to numerical mappings as key, values. This function checks that the dictionary
        was created and is not empty.
        """

        # check that dictionary is not empty
        if len(self.encoder_dict_) == 0:
            raise ValueError(
                "Encoder could not be fitted. Check the parameters and the variables "
                "in your dataframe."
            )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace categories with the learned parameters.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If dataframe is not of same size as that used in fit()
        Warning
            If after encoding, NAN were introduced.

        Returns
        -------
        X : pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing the categories replaced by numbers.
        """

        X = self._check_transform_input_and_state(X)

        # replace categories by the learned parameters
        for feature in self.encoder_dict_.keys():
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if NaN values were introduced by the encoding
        if X[self.encoder_dict_.keys()].isnull().sum().sum() > 0:
            warnings.warn(
                "NaN values were introduced in the returned dataframe by the encoder."
                "This means that some of the categories in the input dataframe were "
                "not present in the training set used when the fit method was called. "
                "Thus, mappings for those categories do not exist. Try using the "
                "RareLabelCategoricalEncoder to remove infrequent categories before "
                "calling this encoder."
            )

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert the encoded variable back to the original values.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the dataframe is not of same size as that used in fit()

        Returns
        -------
        X : pandas dataframe of shape = [n_samples, n_features].
            The un-transformed dataframe, with the categorical variables containing the
            original values.
        """

        X = self._check_transform_input_and_state(X)

        # replace encoded categories by the original values
        for feature in self.encoder_dict_.keys():
            inv_map = {v: k for k, v in self.encoder_dict_[feature].items()}
            X[feature] = X[feature].map(inv_map)

        return X
