# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.dataframe_checks import check_X
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
class Winsorizer(WinsorizerBase):
    """
    The Winsorizer() caps maximum and/or minimum values of a variable at automatically
    determined values, and optionally adds indicators.

    {intro_docstring}

    The Winsorizer() works only with numerical variables. A list of variables can
    be indicated. Alternatively, the Winsorizer() will select and cap all numerical
    variables in the train set.

    The transformer first finds the values at one or both tails of the distributions
    (fit). The transformer then caps the variables (transform).

    More details in the :ref:`User Guide <winsorizer>`.

    Parameters
    ----------
    {capping_method}

    {tail}

    {fold}

    add_indicators: bool, default=False
        Whether to add indicator variables to flag the capped outliers.
        If 'True', binary variables will be added to flag outliers on the left and right
        tails of the distribution. One binary variable per tail, per variable.

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
        Learn the values that will replace the outliers.

    {fit_transform}

    transform:
        Cap the variables.

    """

    def __init__(
        self,
        capping_method: str = "gaussian",
        tail: str = "right",
        fold: Union[int, float] = 3,
        add_indicators: bool = False,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
    ) -> None:
        if not isinstance(add_indicators, bool):
            raise ValueError(
                "add_indicators takes only booleans True and False"
                f"Got {add_indicators} instead."
            )
        super().__init__(capping_method, tail, fold, variables, missing_values)
        self.add_indicators = add_indicators

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cap the variable values. Optionally, add outlier indicators.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features + n_ind]
            The dataframe with the capped variables and indicators.
            The number of output variables depends on the values for 'tail' and
            'add_indicators': if passing 'add_indicators=False', will be equal
            to 'n_features', otherwise, will have an additional indicator column
            per processed feature for each tail.
        """
        if not self.add_indicators:
            X_out = super().transform(X)

        else:
            X_orig = check_X(X)
            X_out = super().transform(X_orig)
            X_orig = X_orig[self.variables_]
            X_out_filtered = X_out[self.variables_]

            if self.tail in ["left", "both"]:
                X_left = X_out_filtered > X_orig
                X_left.columns = [str(cl) + "_left" for cl in self.variables_]
            if self.tail in ["right", "both"]:
                X_right = X_out_filtered < X_orig
                X_right.columns = [str(cl) + "_right" for cl in self.variables_]
            if self.tail == "left":
                X_out = pd.concat([X_out, X_left.astype(np.float64)], axis=1)
            elif self.tail == "right":
                X_out = pd.concat([X_out, X_right.astype(np.float64)], axis=1)
            else:
                X_both = pd.concat([X_left, X_right], axis=1).astype(np.float64)
                X_both = X_both[
                    [
                        cl1
                        for cl2 in zip(X_left.columns.values, X_right.columns.values)
                        for cl1 in cl2
                    ]
                ]
                X_out = pd.concat([X_out, X_both], axis=1)

        return X_out

    def get_feature_names_out(
            self, input_features: Optional[List[Union[str, int]]] = None
    ) -> List[Union[str, int]]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: str, list, default=None
            If `None`, then the names of all the variables in the transformed dataset
            is returned. If list with feature names, the features in the list, plus
            the outlier indicators (if added), will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        feature_names = super().get_feature_names_out(input_features)

        if self.add_indicators is True:

            if input_features is None:
                features = self.variables_
            else:
                features = input_features

            if self.tail == "left":
                indicators = [str(cl) + "_left" for cl in features]
            elif self.tail == "right":
                indicators = [str(cl) + "_right" for cl in features]
            else:
                indicators = []
                for cl in features:
                    indicators.append(str(cl) + "_left")
                    indicators.append(str(cl) + "_right")

            feature_names = feature_names + indicators

        return feature_names
