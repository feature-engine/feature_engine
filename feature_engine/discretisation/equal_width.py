# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _binner_dict_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
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
    return_empty=_return_empty_docstring,
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

    {return_empty}

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
        International Journal of Computer Science, vol. 1, pp. 111-117, 2006.

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
        return_empty: bool = False,
        bins: int = 10,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
    ) -> None:

        if not isinstance(bins, int):
            raise ValueError(f"bins must be an integer. Got {bins} instead.")

        _check_return_empty_is_bool(return_empty)

        super().__init__(return_object, return_boundaries, precision)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.bins = bins

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learn the boundaries of the equal width intervals / bins for each
        variable.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the variables
            to be transformed.
        y: None
            y is not needed in this encoder. You can pass y or None.
        """

        # check input dataframe
        X, variables_ = self._fit_setup(X)

        # fit
        binner_dict_ = {}

        if len(variables_) > 0:
            # one narwhals call for every variable at once, instead of a
            # get_column() round-trip per variable.
            arr = nw.from_native(X, eager_only=True).select(variables_).to_numpy()
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            for var, mn, mx in zip(variables_, mins, maxs):
                binner_dict_[var] = _equal_width_edges(mn, mx, self.bins)

        self.binner_dict_ = binner_dict_
        self.variables_ = variables_
        self._get_feature_names_in(X)

        return self


def _equal_width_edges(mn: float, mx: float, bins: int) -> List[float]:
    """Bin-edge computation matching pandas.cut(bins=int, duplicates="drop"):
    widen a constant [mn, mx] by 0.1% so linspace still produces positive-
    width bins, then collapse duplicate edges the same way. The outer edges
    are then clipped to +-inf, same as the pre-migration code did to the
    retbins output, so transform() never needs an out-of-range branch.
    """
    if mn == mx:
        mn = mn - 0.001 * abs(mn) if mn != 0 else -0.001
        mx = mx + 0.001 * abs(mx) if mx != 0 else 0.001

    edges = np.linspace(mn, mx, bins + 1)
    unique_edges = np.unique(edges)
    if len(unique_edges) < len(edges) and len(edges) != 2:
        edges = unique_edges

    edges_: List[float] = edges.tolist()
    edges_[0] = float("-inf")
    edges_[-1] = float("inf")
    return edges_
