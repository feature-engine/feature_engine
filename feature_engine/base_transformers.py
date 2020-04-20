# Transformation methods are shared by most transformer groups. Each transformer can inherit
# the transform method from these base classes.

import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from feature_engine.utils import _is_dataframe, _check_input_matches_training_df, _check_contains_na
from feature_engine.utils import _find_numerical_variables


class BaseImputer(BaseEstimator, TransformerMixin):
    # Common transformation procedure for most feature imputers
    def transform(self, X):
        """
        Replaces missing data with the learned parameters.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the selected variables
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input data contains same number of columns than the dataframe
        # used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        # replaces missing data with the learned parameters
        for variable in self.imputer_dict_:
            X[variable].fillna(self.imputer_dict_[variable], inplace=True)

        return X


class BaseCategoricalTransformer(BaseEstimator, TransformerMixin):
    # Common transformation procedure for most variable encoders
    def transform(self, X):
        """ Replaces categories with the learned parameters.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features].
            The input samples.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing categories replaced by numbers.
       """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns than the dataframe
        # used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        # replace categories by the learned parameters
        for feature in self.encoder_dict_.keys():
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if NaN values were introduced by the encoding
        if X[self.encoder_dict_.keys()].isnull().sum().sum() > 0:
            warnings.warn(
                "NaN values were introduced in the returned dataframe by the encoder.This means that some "
                "of the categories in the input dataframe were not present in the training set used when  "
                "the fit method was called. Thus mappings for those categories does not exist. "
                "Try using the RareLabelCategoricalEncoder to remove infrequent categories before calling "
                "this encoder.")

        return X


class BaseNumericalTransformer(BaseEstimator, TransformerMixin):
    # shared set-up procedures across numerical transformers, i.e.,
    # variable transformers, discretisers, outlier handlers
    def fit(self, X, y=None):
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        return X

    def transform(self, X):
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns than the dataframe
        # used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X

# class BaseDiscretiser(BaseNumericalTransformer):
#
#     def transform(self, X):
#         """
#         Discretises the variables, that is, sorts the variable values into
#         the learned intervals.
#
#         Parameters
#         ----------
#
#         X : pandas dataframe of shape = [n_samples, n_features]
#             The input samples.
#
#         Returns
#         -------
#
#         X_transformed : pandas dataframe of shape = [n_samples, n_features]
#             The dataframe with discrete / binned variables
#         """
#
#         # Check is fit had been called
#         check_is_fitted(self, ['binner_dict_'])
#
#         if X.shape[1] != self.input_shape_[1]:
#             raise ValueError(
#                 'Number of columns in the dataset is different from training set used to fit the discretiser')
#
#         X = X.copy()
#         for feature in self.variables:
#             X[feature] = pd.cut(X[feature], self.binner_dict_[feature], labels=False)
#
#         if self.return_object:
#             X[self.variables] = X[self.variables].astype('O')
#
#         return X
#
#
# class BaseOutlierRemover(BaseNumericalTransformer):
#
#     def transform(self, X):
#         """
#         Caps variables at the calculated or given parameters.
#
#         Parameters
#         ----------
#
#         X : pandas dataframe of shape = [n_samples, n_features]
#             The input samples.
#
#         Returns
#         -------
#
#         X_transformed : pandas dataframe of shape = [n_samples, n_features]
#             The dataframe with capped values for the selected
#             variables
#         """
#
#         # Check is fit had been called
#         check_is_fitted(self, ['left_tail_caps_', 'right_tail_caps_'])
#
#         if X.shape[1] != self.input_shape_[1]:
#             raise ValueError('Number of columns in dataset is different from training set used to fit the transformer')
#
#         X = X.copy()
#         for feature in self.right_tail_caps_.keys():
#             X[feature] = np.where(X[feature] > self.right_tail_caps_[feature], self.right_tail_caps_[feature],
#                                   X[feature])
#
#         for feature in self.left_tail_caps_.keys():
#             X[feature] = np.where(X[feature] < self.left_tail_caps_[feature], self.left_tail_caps_[feature], X[feature])
#
#         return X
