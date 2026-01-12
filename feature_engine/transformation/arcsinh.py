# Authors: Ankit Hemant Lade (contributor)
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

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
    _fit_not_learn_docstring,
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
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class ArcSinhTransformer(BaseNumericalTransformer):
    """
    The ArcSinhTransformer() applies the inverse hyperbolic sine transformation
    (arcsinh) to numerical variables. Also known as the pseudo-logarithm, this
    transformation is useful for data that contains both positive and negative values.

    The transformation is: x → arcsinh((x - loc) / scale)

    For large values of x, arcsinh(x) behaves like ln(x) + ln(2), providing similar
    variance-stabilizing properties as the log transformation. For small values of x,
    it behaves approximately linearly (i.e., arcsinh(x) ≈ x). This makes it ideal for
    variables like net worth, profit/loss, or any metric that can be positive or
    negative.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    More details in the :ref:`User Guide <arcsinh_transformer>`.

    Parameters
    ----------
    {variables}

    loc: float, default=0.0
        Location parameter for shifting the data before transformation.
        The transformation becomes: arcsinh((x - loc) / scale)

    scale: float, default=1.0
        Scale parameter for normalizing the data before transformation.
        Must be greater than 0. The transformation becomes: arcsinh((x - loc) / scale)

    Attributes
    ----------
    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {inverse_transform}

    transform:
        Transform the variables using the arcsinh function.

    See Also
    --------
    feature_engine.transformation.LogTransformer :
        Applies log transformation (only for positive values).
    feature_engine.transformation.YeoJohnsonTransformer :
        Applies Yeo-Johnson transformation.

    References
    ----------
    .. [1] Burbidge, J. B., Magee, L., & Robb, A. L. (1988). Alternative
           transformations to handle extreme values of the dependent variable.
           Journal of the American Statistical Association, 83(401), 123-127.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.transformation import ArcSinhTransformer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.randn(100) * 1000))
    >>> ast = ArcSinhTransformer()
    >>> ast.fit(X)
    >>> X = ast.transform(X)
    >>> X.head()
              x
    0  7.516076
    1 -6.330816
    2  7.780254
    3  8.825252
    4 -6.995893
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:

        if not isinstance(loc, (int, float)):
            raise ValueError(
                f"loc must be a number (int or float). "
                f"Got {type(loc).__name__} instead."
            )

        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError(
                f"scale must be a positive number (> 0). Got {scale} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.loc = float(loc)
        self.scale = float(scale)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Selects the numerical variables and stores feature names.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Returns
        -------
        self: ArcSinhTransformer
            The fitted transformer.
        """

        # check input dataframe and find/check numerical variables
        X = super().fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the variables using the arcsinh function.

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

        # Ensure float dtype for the transformation
        X[self.variables_] = X[self.variables_].astype(float)

        # Apply arcsinh transformation: arcsinh((x - loc) / scale)
        X.loc[:, self.variables_] = np.arcsinh(
            (X.loc[:, self.variables_] - self.loc) / self.scale
        )

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
        X_tr: pandas dataframe
            The dataframe with the inverse transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        # Inverse transform: x = sinh(y) * scale + loc
        X.loc[:, self.variables_] = (
            np.sinh(X.loc[:, self.variables_]) * self.scale + self.loc
        )

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
