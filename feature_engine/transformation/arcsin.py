# Authors: Tommaso Pellegrino <tommasopellegrino.1995@gmail.com>
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
class ArcsinTransformer(BaseNumericalTransformer):
    """
    The ArcsinTransformer() applies the arcsin transformation to numerical variables.

    The arcsin transformation, also called arcsin square root transformation, or
    angular transformation, takes the form of arcsin(sqrt(x)) where x is a real number
    between 0 and 1.

    The arcsin square root transformation helps in dealing with probabilities,
    percents, and proportions. It aims to stabilize the variance of the variable and
    return more evenly distributed (Gaussian looking) values.

    The ArcsinTransformer() only works with numerical variables which values are
    between 0 and 1. If a variable contains values outside of this range, the
    transformer will raise an error.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all numerical variables.

    More details in the :ref:`User Guide <arcsin>`.

    Parameters
    ----------
    {variables}

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
        Apply the arcsin transformation.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.transformation import ArcsinTransformer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.beta(1, 1, size = 100)))
    >>> ast = ArcsinTransformer()
    >>> ast.fit(X)
    >>> X = ast.transform(X)
    >>> X.head()
              x
    0  0.785437
    1  0.253389
    2  0.144664
    3  0.783236
    4  0.650777
    """

    def __init__(
        self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:

        self.variables = _check_variables_input_value(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = super().fit(X)

        # check if the variables are in the correct range
        if ((X[self.variables_] < 0) | (X[self.variables_] > 1)).any().any():
            raise ValueError(
                "Some variables contain values outside the possible range 0-1. "
                "Can't apply the arcsin transformation. "
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the arcsin transformation.

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

        # check if the variables are in the correct range
        if ((X[self.variables_] < 0) | (X[self.variables_] > 1)).any().any():
            raise ValueError(
                "Some variables contain values outside the possible range 0-1. "
                "Can't apply the arcsin transformation."
            )

        # transform
        X.loc[:, self.variables_] = np.arcsin(np.sqrt(X.loc[:, self.variables_]))

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
        # inverse_transform
        X.loc[:, self.variables_] = (np.sin(X.loc[:, self.variables_])) ** 2

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # =======  this tests fail because the transformers throw an error when the
        # values are less than 0 or greater than 1. Nothing to do with the test itself
        # but mostly with the data created and used in the test
        msg = (
            "transformers raise errors when data is outside [0, 1] range, thus this"
            "check fails"
        )
        tags_dict["_xfail_checks"]["check_estimators_dtypes"] = msg
        tags_dict["_xfail_checks"]["check_estimators_fit_returns_self"] = msg
        tags_dict["_xfail_checks"]["check_pipeline_consistency"] = msg
        tags_dict["_xfail_checks"]["check_estimators_overwrite_params"] = msg
        tags_dict["_xfail_checks"]["check_estimators_pickle"] = msg
        tags_dict["_xfail_checks"]["check_transformer_general"] = msg
        tags_dict["_xfail_checks"]["check_methods_subset_invariance"] = msg
        tags_dict["_xfail_checks"]["check_fit2d_1sample"] = msg
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg
        tags_dict["_xfail_checks"]["check_dict_unchanged"] = msg
        tags_dict["_xfail_checks"]["check_dont_overwrite_parameters"] = msg
        tags_dict["_xfail_checks"]["check_fit_check_is_fitted"] = msg
        tags_dict["_xfail_checks"]["check_n_features_in"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        return super().__sklearn_tags__()
