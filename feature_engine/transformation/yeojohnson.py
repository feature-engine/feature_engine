# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import narwhals as nw
import numpy as np
import scipy.stats as stats
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.tags import _return_tags


@Substitution(
    variables=_variables_numerical_docstring,
    return_empty=_return_empty_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class YeoJohnsonTransformer(BaseNumericalTransformer):
    """
    The YeoJohnsonTransformer() applies the Yeo-Johnson transformation to the
    numerical variables.

    The Yeo-Johnson transformation implemented by this transformer is that of
    SciPy.stats:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html

    The YeoJohnsonTransformer() works only with numerical variables.

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.

    More details in the :ref:`User Guide <yeojohnson>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    Attributes
    ----------
    lambda_dict_
        Dictionary containing the best lambda for the Yeo-Johnson per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the optimal lambda for the Yeo-Johnson transformation.

    {fit_transform}

    {inverse_transform}

    transform:
        Apply the Yeo-Johnson transformation.

    References
    ----------
    .. [1] Yeo, In-Kwon and Johnson, Richard (2000).
        A new family of power transformations to improve normality or symmetry.
        Biometrika, 87, 954-959.

    .. [2] Weisberg S. "Yeo-Johnson Power Transformations".
        https://www.stat.umn.edu/arc/yjpower.pdf

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.transformation import YeoJohnsonTransformer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100) - 10))
    >>> yjt = YeoJohnsonTransformer()
    >>> yjt.fit(X)
    >>> X = yjt.transform(X)
    >>> X.head()
                   x
    0 -267042.661354
    1 -444356.715596
    2 -221625.915167
    3  -23647.614887
    4 -467264.546413

    With polars:

    >>> import numpy as np
    >>> import polars as pl
    >>> from feature_engine.transformation import YeoJohnsonTransformer
    >>> np.random.seed(42)
    >>> X = pl.DataFrame({"x": list(np.random.lognormal(size=6) - 10)})
    >>> yjt = YeoJohnsonTransformer()
    >>> yjt.fit(X)
    >>> yjt.transform(X)
    shape: (6, 1)
    ┌────────────────┐
    │ x              │
    │ ---            │
    │ f64            │
    ╞════════════════╡
    │ -467714.164249 │
    │ -795057.401919 │
    │ -385148.281012 │
    │ -37417.353351  │
    │ -837807.71099  │
    │ -837800.580457 │
    └────────────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:
        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learn the optimal lambda for the Yeo-Johnson transformation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X, variables_ = self._fit_setup(X)

        values = nw.from_native(X, eager_only=True).select(variables_).to_numpy()
        values = values.astype(float)

        # scipy searches the optimal lambda one column at a time, there is no
        # vectorized multi-column form of the search.
        lambda_dict_ = {}
        for i, var in enumerate(variables_):
            _, lambda_dict_[var] = stats.yeojohnson(values[:, i])

        self.variables_ = variables_
        self.lambda_dict_ = lambda_dict_
        self._get_feature_names_in(X)

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Apply the Yeo-Johnson transformation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        nw_X = nw.from_native(X, eager_only=True)
        values = nw_X.select(self.variables_).to_numpy().astype(float)

        # transform
        result = np.empty_like(values)
        for i, var in enumerate(self.variables_):
            result[:, i] = stats.yeojohnson(values[:, i], lmbda=self.lambda_dict_[var])

        new_series = [
            nw.new_series(var, result[:, i], backend=nw_X.implementation)
            for i, var in enumerate(self.variables_)
        ]
        X = nw_X.with_columns(*new_series).to_native()

        return X

    def inverse_transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_tr: dataframe
            The dataframe with the transformed variables.
        """
        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        nw_X = nw.from_native(X, eager_only=True)
        values = nw_X.select(self.variables_).to_numpy().astype(float)

        # inverse_transform
        result = np.empty_like(values)
        for i, var in enumerate(self.variables_):
            result[:, i] = self._inverse_transform_array(
                values[:, i], lmbda=self.lambda_dict_[var]
            )

        new_series = [
            nw.new_series(var, result[:, i], backend=nw_X.implementation)
            for i, var in enumerate(self.variables_)
        ]
        X = nw_X.with_columns(*new_series).to_native()

        return X

    def _inverse_transform_array(self, X: np.ndarray, lmbda: float) -> np.ndarray:
        x_inv = np.zeros_like(X)
        pos = X >= 0

        # when x >= 0
        if lmbda == 0:
            x_inv[pos] = np.exp(X[pos]) - 1
        else:  # lmbda != 0
            x_inv[pos] = np.power(X[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if lmbda != 2:
            x_inv[~pos] = 1 - np.power(-(2 - lmbda) * X[~pos] + 1, 1 / (2 - lmbda))
        else:  # lmbda == 2
            x_inv[~pos] = 1 - np.exp(-X[~pos])

        return x_inv

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"

        # =======  this tests fail because the transformers throw an error
        # when the values are 0. Nothing to do with the test itself but
        # mostly with the data created and used in the test
        msg = (
            "Transformer raises error when it can't find the optimal lambda for "
            "the transformation, thus this check fails."
        )
        tags_dict["_xfail_checks"]["check_fit2d_1sample"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        return super().__sklearn_tags__()
