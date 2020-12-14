# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, Dict

import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer


class ArbitraryDiscretiser(BaseNumericalTransformer):
    """
    The ArbitraryDiscretiser() divides continuous numerical variables into contiguous
    intervals which limits are determined arbitrarily by the user.

    The user needs to enter a dictionary with variable names as keys, and a list of
    the limits of the intervals as values. For example {'var1':[0, 10, 100, 1000],
    'var2':[5, 10, 15, 20]}.

    ArbitraryDiscretiser() will then sort var1 values into the intervals 0-10, 10-100
    100-1000, and var2 into 5-10, 10-15 and 15-20. Similar to `pandas.cut`.

    The  ArbitraryDiscretiser() works only with numerical variables. The discretiser
    will check if the dictionary entered by the user contains variables present in the
    training set, and if these variables are numerical, before doing any
    transformation.

    Then it transforms the variables, that is, it sorts the values into the intervals.

    Parameters
    ----------
    binning_dict : dict
        The dictionary with the variable : interval limits pairs, provided by the user.
        A valid dictionary looks like this:

        binning_dict = {'var1':[0, 10, 100, 1000], 'var2':[5, 10, 15, 20]}.

    return_object : bool, default=False
        Whether the numbers in the discrete variable should be returned as
        numeric or as object. The decision is made by the user based on
        whether they would like to proceed the engineering of the variable as
        if it was numerical or categorical.

        Categorical encoders in Feature-engine work only with variables of type object,
        thus, if you wish to encode the returned bins, set return_object to True.

    return_boundaries : bool, default=False
        whether the output, that is the bin names / values, should be the interval
        boundaries. If True, it returns the interval boundaries. If False, it returns
        integers.

    Attributes
    ----------
    binner_dict_ :
         Dictionary with the interval limits per variable.

    Methods
    -------
    fit:
        This transformer does not learn any parameter.
    transform:
        Sort continuous variable values into the intervals.
    fit_transform:
        Fit to the data, then transform it.

    See Also
    --------
    pandas.cut :
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html

    """

    def __init__(
        self,
        binning_dict: Dict[str, list],
        return_object: bool = False,
        return_boundaries: bool = False,
    ) -> None:

        if not isinstance(binning_dict, dict):
            raise ValueError(
                "Please provide at a dictionary with the interval limits per variable"
            )

        if not isinstance(return_object, bool):
            raise ValueError("return_object must be True or False")

        self.binning_dict = binning_dict
        self.variables = [x for x in binning_dict.keys()]
        self.return_object = return_object
        self.return_boundaries = return_boundaries

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Check dataframe and variables. Checks that the user entered variables are in
        the train set and cast as numerical.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y : None
            y is not needed in this encoder. You can pass y or None.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            - If there are no numerical variables in the df or the df is empty
            - If the variable(s) contain null values

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X, y)

        # check that all variables in the dictionary are present in the df
        if all(variable in X.columns for variable in self.variables):
            self.binner_dict_ = self.binning_dict
        else:
            raise ValueError(
                "There are variables in the provided dictionary which are not present "
                "in the train set or not cast as numerical"
            )

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sort the variable values into the intervals.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Raises
        ------
        TypeError
           If the input is not a Pandas DataFrame
        ValueError
           - If the variable(s) contain null values
           - If the dataframe is not of the same size as the one used in fit()

        Returns
        -------
        X : pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform variables
        if self.return_boundaries:
            for feature in self.variables:
                X[feature] = pd.cut(
                    X[feature], self.binner_dict_[feature]  # type: ignore
                )

        else:
            for feature in self.variables:
                X[feature] = pd.cut(
                    X[feature], self.binner_dict_[feature], labels=False  # type: ignore
                )

            # return object
            if self.return_object:
                X[self.variables] = X[self.variables].astype("O")

        return X
