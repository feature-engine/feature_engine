# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd

from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import (
    _variables_numerical_docstring,
    _missing_values_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.outliers.base_outlier import WinsorizerBase


@Substitution(
    intro_docstring=WinsorizerBase._intro_docstring,
    capping_method=WinsorizerBase._capping_method_docstring,
    tail=WinsorizerBase._tail_docstring,
    fold=WinsorizerBase._fold_docstring,
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    right_tail_caps_=WinsorizerBase._right_tail_caps_docstring,
    left_tail_caps_=WinsorizerBase._left_tail_caps_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class OutlierTrimmer(WinsorizerBase):
    """The OutlierTrimmer() removes observations with outliers from the dataset.

    The OutlierTrimmer() first calculates the maximum and /or minimum values
    beyond which a value will be considered an outlier, and thus removed.

    {intro_docstring}

    The OutlierTrimmer() works only with numerical variables. A list of variables can
    be indicated. Alternatively, it will select all numerical variables.

    The transformer first finds the values at one or both tails of the distributions
    (fit). The transformer then removes observations with outliers from the dataframe
    (transform).

    More details in the :ref:`User Guide <outlier_trimmer>`.

    Parameters
    ----------
    {capping_method}

    {tail}

    {fold}

    {variables}

    {missing_values}

    Attributes
    ----------
    {right_tail_caps_}

    {left_tail_caps_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find maximum and minimum values.

    {fit_transform}

    transform:
        Remove outliers.

    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove observations with outliers from the dataframe.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe without outlier observations.
        """

        X = self._check_transform_input_and_state(X)

        for feature in self.right_tail_caps_.keys():
            outliers = np.where(
                X[feature] > self.right_tail_caps_[feature], True, False
            )
            X = X.loc[~outliers]

        for feature in self.left_tail_caps_.keys():
            outliers = np.where(X[feature] < self.left_tail_caps_[feature], True, False)
            X = X.loc[~outliers]

        return X
