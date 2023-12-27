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
class EqualFrequencyDiscretiser(BaseDiscretiser):
    """
    The EqualFrequencyDiscretiser() divides continuous numerical variables
    into contiguous equal frequency intervals, that is, intervals that contain
    approximately the same proportion of observations.

    The EqualFrequencyDiscretiser() works only with numerical variables.
    A list of variables can be passed as argument. Alternatively, the discretiser
    will automatically select and transform all numerical variables.

    The EqualFrequencyDiscretiser() first finds the boundaries for the intervals or
    quantiles for each variable. Then it transforms the variables, that is, it sorts
    the values into the intervals.

    More details in the :ref:`User Guide <equal_freq_discretiser>`.

    Parameters
    ----------
    {variables}

    q: int, default=10
        Desired number of equal frequency intervals / bins.

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
    pandas.qcut
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
    >>> from feature_engine.discretisation import EqualFrequencyDiscretiser
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.randint(1,100, 100)))
    >>> efd = EqualFrequencyDiscretiser()
    >>> efd.fit(X)
    >>> efd.transform(X)["x"].value_counts()
    8    12
    6    11
    3    11
    1    10
    5    10
    2    10
    0    10
    4     9
    7     9
    9     8
    Name: x, dtype: int64
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        q: int = 10,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
    ) -> None:

        if not isinstance(q, int):
            raise ValueError(f"q must be an integer. Got {q} instead.")

        super().__init__(return_object, return_boundaries, precision)

        self.q = q
        self.variables = _check_variables_input_value(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the limits of the equal frequency intervals.

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

        self.binner_dict_ = {}

        for var in self.variables_:
            tmp, bins = pd.qcut(x=X[var], q=self.q, retbins=True, duplicates="drop")

            # Prepend/Append infinities to accommodate outliers
            bins = list(bins)
            bins[0] = float("-inf")
            bins[len(bins) - 1] = float("inf")
            self.binner_dict_[var] = bins

        return self
