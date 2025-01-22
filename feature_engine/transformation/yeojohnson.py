# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

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
    0 -267042.906453
    1 -444357.138990
    2 -221626.115742
    3  -23647.632651
    4 -467264.993249
    """

    def __init__(
        self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:
        self.variables = _check_variables_input_value(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the optimal lambda for the Yeo-Johnson transformation.

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
            _, self.lambda_dict_[var] = stats.yeojohnson(X[var])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Yeo-Johnson transformation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X: pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted

        X = self._check_transform_input_and_state(X)
        for feature in self.variables_:
            X[feature] = stats.yeojohnson(X[feature], lmbda=self.lambda_dict_[feature])

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_tr: pandas dataframe
            The dataframe with the transformed variables.
        """
        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        for feature in self.variables_:
            X[feature] = self._inverse_transform_series(
                X[feature], lmbda=self.lambda_dict_[feature]
            )

        return X

    def _inverse_transform_series(self, X: pd.Series, lmbda: float) -> pd.Series:
        x_inv = pd.Series(np.zeros_like(X), index=X.index)
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
