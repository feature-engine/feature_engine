import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)
from feature_engine.variable_manipulation import _find_all_variables, _define_variables


class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):

    """
    DropCorrelatedFeatures finds and removes correlated features

    Parameters
    ----------
    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.

    method: string, default='pearson'
    methods of correlation:
        pearson : standard correlation coefficient
        kendall : Kendall Tau correlation coefficient
        spearman : Spearman rank correlation
    Threshold: float, default=1.0
    """

    def __init__(self, variables=None, method='pearson', threshold=1.0):
        self.variables = _define_variables(variables)
        self.method = method
        self.threshold = threshold
        # create a set where I will store the names of correlated columns
        self.col_corr = set()
        self.corr_matrix = pd.DataFrame()

    # with the following function we can select highly correlated features
        # it will remove the first feature that is correlated with anything else
        # without any further insight.

    # def correlation(dataset, threshold):
    def fit(self, X, y=None):

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are in the dataframe
        self.variables = _find_all_variables(X, self.variables)

        # check that X is numeric
        if X.shape[1] != X.select_dtypes(include=np.number).shape[1]:
            raise TypeError(
                "This transformer is only designed for numeric dataframes."
            )

        # create the correlation matrix
        self.corr_matrix = X.corr(self.method)

        # for each feature in the dataset (columns of the correlation matrix)
        for i in range(len(self.corr_matrix.columns)):

            # check with other features
            for j in range(i):

                # if the correlation is higher than a certain threshold
                if abs(self.corr_matrix.iloc[i, j]) > self.threshold:  # we are interested in absolute coeff value

                    # get the name of the correlated feature and add it to our correlated set
                    self.col_corr.add(self.corr_matrix.columns[i])

        return self.col_corr

    def transform(self, X):
        """
        Drops the correlated features from a dataframe.
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input samples.
        Returns
        -------
        X_transformed: pandas dataframe,
            shape = [n_samples, n_features - (duplicated features)]
            The transformed dataframe with the remaining subset of variables.
        """
        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.col_corr.count())

        # returned non-duplicate features
        X = X.drop(columns=self.col_corr)

        return X
