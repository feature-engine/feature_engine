# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine.variable_manipulation import _check_input_parameter_variables
from feature_engine.discretisation.base_discretiser import BaseDiscretiser


class EqualWidthDiscretiser(BaseDiscretiser):
    """
    The EqualWidthDiscretiser() divides continuous numerical variables into
    intervals of the same width, that is, equidistant intervals. Note that the
    proportion of observations per interval may vary.

    The size of the interval is calculated as:

    .. math::

        ( max(X) - min(X) ) / bins

    where bins, which is the number of intervals, is determined by the user.

    The EqualWidthDiscretiser() works only with numerical variables.
    A list of variables can be passed as argument. Alternatively, the discretiser
    will automatically select all numerical variables.

    The EqualWidthDiscretiser() first finds the boundaries for the intervals for
    each variable. Then, it transforms the variables, that is, sorts the values into
    the intervals.

    More details in the :ref:`User Guide <equal_width_discretiser>`.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the
        discretiser will automatically select all numerical type variables.

    bins: int, default=10
        Desired number of equal width intervals / bins.

    return_object: bool, default=False
        Whether the the discrete variable should be returned as numeric or as
        object. If you would like to proceed with the engineering of the variable as if
        it was categorical, use True. Alternatively, keep the default to False.

    return_boundaries : bool, default=False
        Whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.

    Attributes
    ----------
    binner_dict_:
        Dictionary with the interval limits per variable.

    variables_:
         The variables that will be discretised.

    n_features_in_:
        The number of features in the train set used in fit.

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
    pandas.cut
    sklearn.preprocessing.KBinsDiscretizer

    References
    ----------
    .. [1] Kotsiantis and Pintelas, "Data preprocessing for supervised leaning,"
        International Journal of Computer Science,  vol. 1, pp. 111 117, 2006.

    .. [2] Dong. "Beating Kaggle the easy way". Master Thesis.
        https://www.ke.tu-darmstadt.de/lehre/arbeiten/studien/2015/Dong_Ying.pdf

    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        bins: int = 10,
    ) -> None:

        if not isinstance(bins, int):
            raise ValueError("q must be an integer")

        super().__init__(return_object, return_boundaries)

        self.bins = bins
        self.variables = _check_input_parameter_variables(variables)


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the boundaries of the equal width intervals / bins for each
        variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the variables
            to be transformed.
        y: None
            y is not needed in this encoder. You can pass y or None.
        """

        # check input dataframe
        X = super().fit(X, y)

        # fit
        self.binner_dict_ = {}

        for var in self.variables_:
            tmp, bins = pd.cut(
                x=X[var], bins=self.bins, retbins=True, duplicates="drop"
            )

            # Prepend/Append infinities
            bins = list(bins)
            bins[0] = float("-inf")
            bins[len(bins) - 1] = float("inf")
            self.binner_dict_[var] = bins

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the variable values into the intervals.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """

        # transform variables
        X = super().transform(X)

        return X
