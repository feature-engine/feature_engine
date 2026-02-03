# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd
import scipy.stats as stats
from scipy.special import inv_boxcox

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
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
    """

    def __init__(
        self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:

        self.variables = _check_variables_input_value(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the optimal lambda for the BoxCox transformation.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = super().fit(X)

        self.lambda_dict_ = {}

        for var in self.variables_:
            _, self.lambda_dict_[var] = stats.boxcox(X[var])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the BoxCox transformation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        # check contains zero or negative values
        if (X[self.variables_] <= 0).any().any():
            raise ValueError("Data must be positive.")

        # transform
        for feature in self.variables_:
            X[feature] = stats.boxcox(X[feature], lmbda=self.lambda_dict_[feature])

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be inverse transformed.

        Returns
        -------
        X_new: pandas dataframe
            The dataframe with the original variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        # inverse transform
        for feature in self.variables_:
            X[feature] = inv_boxcox(X[feature], self.lambda_dict_[feature])

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
