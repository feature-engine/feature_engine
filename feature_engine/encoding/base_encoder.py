import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _is_dataframe, _check_contains_na, _check_input_matches_training_df


def _check_encoding_dictionary(dictionary):
    # check that there is a dictionary with category to number pairs
    if len(dictionary) == 0:
        raise ValueError('Encoder could not be fitted. Check the parameters and the variables '
                         'in your dataframe.')
    return dictionary


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

        # Check that input data contains same number of columns as dataframe used to fit
        _check_input_matches_training_df(X, self.input_shape_[1])

        # replace categories by the learned parameters
        for feature in self.encoder_dict_.keys():
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if NaN values were introduced by the encoding
        if X[self.encoder_dict_.keys()].isnull().sum().sum() > 0:
            warnings.warn(
                "NaN values were introduced in the returned dataframe by the encoder."
                "This means that some of the categories in the input dataframe were "
                "not present in the training set used when the fit method was called. "
                "Thus, mappings for those categories does not exist. Try using the "
                "RareLabelCategoricalEncoder to remove infrequent categories before "
                "calling this encoder."
            )

        return X

    # restores encoded variables to their original values
    def inverse_transform(self, X):
        """ Convert the data back to the original representation.

        Parameters
        ----------

        X_transformed : pandas dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------

        X : pandas dataframe of shape = [n_samples, n_features].
            The un-transformed dataframe, that is, containing the original values
            of the categorical variables.
       """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # Check that input data contains same number of columns as dataframe used to fit
        _check_input_matches_training_df(X, self.input_shape_[1])

        # replace encoded categories by the original values
        for feature in self.encoder_dict_.keys():
            inv_map = {v: k for k, v in self.encoder_dict_[feature].items()}
            X[feature] = X[feature].map(inv_map)

        return X
