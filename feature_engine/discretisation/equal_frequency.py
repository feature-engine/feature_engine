# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd
from feature_engine.variable_manipulation import _check_input_parameter_variables
from feature_engine.base_transformers import BaseNumericalTransformer


class EqualFrequencyDiscretiser(BaseNumericalTransformer):
    """
    The EqualFrequencyDiscretiser() divides continuous numerical variables
    into contiguous equal frequency intervals, that is, intervals that contain
    approximately the same proportion of observations.

    The interval limits are determined using `pandas.qcut()`, in other words,
    the interval limits are determined by the quantiles. The number of intervals,
    i.e., the number of quantiles in which the variable should be divided is
    determined by the user.

    The EqualFrequencyDiscretiser() works only with numerical variables.
    A list of variables can be passed as argument. Alternatively, the discretiser
    will automatically select and transform all numerical variables.

    The EqualFrequencyDiscretiser() first finds the boundaries for the intervals or
    quantiles for each variable.

    Then it transforms the variables, that is, it sorts the values into the intervals.

    Parameters
    ----------
    q : int, default=10
        Desired number of equal frequency intervals / bins. In other words the
        number of quantiles in which the variables should be divided.

    variables : list
        The list of numerical variables that will be discretised. If None, the
        EqualFrequencyDiscretiser() will select all numerical variables.

    return_object : bool, default=False
        Whether the numbers in the discrete variable should be returned as
        numeric or as object. The decision is made by the user based on
        whether they would like to proceed the engineering of the variable as
        if it was numerical or categorical.

    return_boundaries : bool, default=False
        whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.

    Attributes
    ----------
    binner_dict_:
         Dictionary with the interval limits per variable.

    Methods
    -------
    fit:
        Find the interval limits.
    transform:
        Sort continuous variable values into the intervals.
    fit_transform:
        Fit to the data, then transform it.

    See Also
    --------
    pandas.qcut :
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html

    References
    ----------
    .. [1] Kotsiantis and Pintelas, "Data preprocessing for supervised leaning,"
        International Journal of Computer Science,  vol. 1, pp. 111 117, 2006.

    .. [2] Dong. "Beating Kaggle the easy way". Master Thesis.
        https://www.ke.tu-darmstadt.de/lehre/arbeiten/studien/2015/Dong_Ying.pdf
    """

    def __init__(
        self,
        q: int = 10,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_object: bool = False,
        return_boundaries: bool = False,
    ) -> None:

        if not isinstance(q, int):
            raise ValueError("q must be an integer")

        if not isinstance(return_object, bool):
            raise ValueError("return_object must be True or False")

        if not isinstance(return_boundaries, bool):
            raise ValueError("return_boundaries must be True or False")

        self.q = q
        self.variables = _check_input_parameter_variables(variables)
        self.return_object = return_object
        self.return_boundaries = return_boundaries

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the limits of the equal frequency intervals, that is the percentiles
        for each variable.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the variables
            to be transformed.
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

        self.binner_dict_ = {}

        for var in self.variables:
            tmp, bins = pd.qcut(x=X[var], q=self.q, retbins=True, duplicates="drop")

            # Prepend/Append infinities to accommodate outliers
            bins = list(bins)
            bins[0] = float("-inf")
            bins[len(bins) - 1] = float("inf")
            self.binner_dict_[var] = bins

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sort the variable values into the intervals.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

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
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature])

        else:
            for feature in self.variables:
                X[feature] = pd.cut(
                    X[feature], self.binner_dict_[feature], labels=False
                )

            # return object
            if self.return_object:
                X[self.variables] = X[self.variables].astype("O")

        return X
