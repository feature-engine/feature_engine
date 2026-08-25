# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import narwhals as nw
import numpy as np
import scipy.special as spsp
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
class BoxCoxTransformer(BaseNumericalTransformer):
    """
    The BoxCoxTransformer() applies the BoxCox transformation to numerical
    variables.

    The Box-Cox transformation is defined as:

    - T(Y)=(Y exp(λ)−1)/λ if λ!=0
    - log(Y) otherwise

    where Y is the response variable and λ is the transformation parameter. λ varies,
    typically from -5 to 5. In the transformation, all values of λ are considered and
    the optimal value for a given variable is selected.

    The BoxCox transformation implemented by this transformer is that of
    SciPy.stats:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html

    The BoxCoxTransformer() works only with numerical positive variables (>=0).

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.

    More details in the :ref:`User Guide <box_cox>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    Attributes
    ----------
    lambda_dict_:
        Dictionary with the best BoxCox exponent per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the optimal lambda for the BoxCox transformation.

    {fit_transform}

    {inverse_transform}

    transform:
        Apply the BoxCox transformation.

    References
    ----------
    .. [1] Box and Cox. "An Analysis of Transformations". Read at a RESEARCH MEETING,
        1964.
        https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1964.tb00553.x

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.transformation import BoxCoxTransformer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100)))
    >>> bct = BoxCoxTransformer()
    >>> bct.fit(X)
    >>> X = bct.transform(X)
    >>> X.head()
              x
    0  0.505485
    1 -0.137595
    2  0.662654
    3  1.607518
    4 -0.232237

    With polars:

    >>> import numpy as np
    >>> import polars as pl
    >>> from feature_engine.transformation import BoxCoxTransformer
    >>> np.random.seed(42)
    >>> X = pl.DataFrame({"x": list(np.random.lognormal(size=6))})
    >>> bct = BoxCoxTransformer()
    >>> bct.fit(X)
    >>> bct.transform(X)
    shape: (6, 1)
    ┌───────────┐
    │ x         │
    │ ---       │
    │ f64       │
    ╞═══════════╡
    │ 0.403681  │
    │ -0.146883 │
    │ 0.495725  │
    │ 0.845914  │
    │ -0.259585 │
    │ -0.259565 │
    └───────────┘
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
        Learn the optimal lambda for the BoxCox transformation.

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

        nw_X = nw.from_native(X, eager_only=True)
        values = nw_X.select(variables_).to_numpy().astype(float)

        lambda_dict_ = {}
        # lambda search is per-column and not vectorizable across columns,
        # unlike transform()'s elementwise application once lambdas are known
        for i, var in enumerate(variables_):
            _, lambda_dict_[var] = stats.boxcox(values[:, i])

        self.variables_ = variables_
        self.lambda_dict_ = lambda_dict_
        self._get_feature_names_in(X)

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Apply the BoxCox transformation.

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

        # check contains zero or negative values
        if (values <= 0).any():
            raise ValueError("Data must be positive.")

        # transform
        lmbdas = np.array([self.lambda_dict_[var] for var in self.variables_])
        result = spsp.boxcox(values, lmbdas)
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
            The data to be inverse transformed.

        Returns
        -------
        X_new: dataframe
            The dataframe with the original variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        nw_X = nw.from_native(X, eager_only=True)
        values = nw_X.select(self.variables_).to_numpy().astype(float)

        # inverse transform
        lmbdas = np.array([self.lambda_dict_[var] for var in self.variables_])
        result = spsp.inv_boxcox(values, lmbdas)
        new_series = [
            nw.new_series(var, result[:, i], backend=nw_X.implementation)
            for i, var in enumerate(self.variables_)
        ]
        X = nw_X.with_columns(*new_series).to_native()

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # =======  this tests fail because the transformers throw an error
        # when the values are 0. Nothing to do with the test itself but
        # mostly with the data created and used in the test
        msg = (
            "transformers raise errors when data contains zeroes, thus this check fails"
        )
        tags_dict["_xfail_checks"]["check_estimators_dtypes"] = msg
        tags_dict["_xfail_checks"]["check_estimators_fit_returns_self"] = msg
        tags_dict["_xfail_checks"]["check_pipeline_consistency"] = msg
        tags_dict["_xfail_checks"]["check_estimators_overwrite_params"] = msg
        tags_dict["_xfail_checks"]["check_estimators_pickle"] = msg
        tags_dict["_xfail_checks"]["check_transformer_general"] = msg

        # boxcox fails this test as well
        msg = "scipy.stats.boxcox does not like the input data"
        tags_dict["_xfail_checks"]["check_methods_subset_invariance"] = msg
        tags_dict["_xfail_checks"]["check_fit2d_1sample"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
