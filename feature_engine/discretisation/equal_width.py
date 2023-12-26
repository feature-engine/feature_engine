# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _binner_dict_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _variables_numerical_docstring,
)
from feature_engine._docstrings.init_parameters.discretisers import (
    _precision_docstring,
    _return_boundaries_docstring,
    _return_object_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_discretiser_docstring,
    _fit_transform_docstring,
    _transform_discretiser_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.discretisation.base_discretiser import BaseDiscretiser


@Substitution(
    return_object=_return_object_docstring,
    return_boundaries=_return_boundaries_docstring,
    precision=_precision_docstring,
    binner_dict_=_binner_dict_docstring,
    fit=_fit_discretiser_docstring,
    transform=_transform_discretiser_docstring,
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
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
    {variables}

    bins: int, default=10
        Desired number of equal width intervals / bins.

    {return_object}

    {return_boundaries}

    {precision}

    Attributes
    ----------
    {binner_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

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

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.discretisation import EqualWidthDiscretiser
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.randint(1,100, 100)))
    >>> ewd = EqualWidthDiscretiser()
    >>> ewd.fit(X)
    >>> ewd.transform(X)["x"].value_counts()
    9    15
    6    15
    0    13
    5    11
    8     9
    7     8
    2     8
    1     7
    3     7
    4     7
    Name: x, dtype: int64
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        bins: int = 10,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
    ) -> None:

        if not isinstance(bins, int):
            raise ValueError(f"bins must be an integer. Got {bins} instead.")

        super().__init__(return_object, return_boundaries, precision)

        self.bins = bins
        self.variables = _check_variables_input_value(variables)

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
        X = super().fit(X)

        # fit
        self.binner_dict_ = {}

        for var in self.variables_:
            tmp, bins = pd.cut(
                x=X[var],
                bins=self.bins,
                retbins=True,
                duplicates="drop",
                include_lowest=True,
            )

            # Prepend/Append infinities
            bins = list(bins)
            bins[0] = float("-inf")
            bins[len(bins) - 1] = float("inf")
            self.binner_dict_[var] = bins

        return self
