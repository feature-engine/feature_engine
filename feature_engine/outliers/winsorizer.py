# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Literal, Union

import numpy as np
import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _left_tail_caps_docstring,
    _n_features_in_docstring,
    _right_tail_caps_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.init_parameters.outliers import (
    _capping_method_docstring,
    _fold_docstring,
    _tail_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.outliers.base_outlier import WinsorizerBase


@Substitution(
    intro_docstring=WinsorizerBase._intro_docstring,
    capping_method=_capping_method_docstring,
    tail=_tail_docstring,
    fold=_fold_docstring,
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    right_tail_caps_=_right_tail_caps_docstring,
    left_tail_caps_=_left_tail_caps_docstring,
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

    fold_:
        Factor multiplying the std, mad, iqr or alternative the percentile. Only
        different from `fold` when `fold="auto"`.

    Methods
    -------
    fit:
        Learn the values that will replace the outliers.

    {fit_transform}

    transform:
        Cap the variables.

    References
    ----------
    .. [1] Rousseeuw, Croux. "Alternatives to the mean absolute deviation". Journal of
       the American Statistical Association, 1993. http://www.jstor.org/stable/2291267 .

    .. [2] Leys, et. al. "Do not use standard deviation around the mean, use absolute
       deviation around the median". Journal of Experimental Social Psychology, 2013.
       http://dx.doi.org/10.1016/j.jesp.2013.03.013.

    .. [3] Thériault, et. al. Check your outliers! An introduction to identifying
       statistical outliers in R with easystats. Behavior Research Methods, 2024.
       https://doi.org/10.3758/s13428-024-02356-w

    .. [4] Dixon. Simplified Estimation from Censored Normal Samples. The Annals of
       Mathematical Statistics, 1960. http://www.jstor.org/stable/2237953

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.outliers import Winsorizer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.normal(size = 10)))
    >>> wz = Winsorizer(capping_method='mad', tail='both', fold=3)
    >>> wz.fit(X)
    >>> wz.transform(X)
              x
    0  0.496714
    1 -0.138264
    2  0.647689
    3  1.523030
    4 -0.234153
    5 -0.234137
    6  1.579213
    7  0.767435
    8 -0.469474
    9  0.542560

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.outliers import Winsorizer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.normal(size = 10)))
    >>> wz = Winsorizer(capping_method='mad', tail='both', fold=3)
    >>> wz.fit(X)
    >>> wz.transform(X)
              x
    0  0.496714
    1 -0.138264
    2  0.647689
    3  1.523030
    4 -0.234153
    5 -0.234137
    6  1.579213
    7  0.767435
    8 -0.469474
    9  0.542560
    """

    def __init__(
        self,
        capping_method: str = "gaussian",
        tail: str = "right",
        fold: Union[int, float, Literal["auto"]] = "auto",
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
            X_out = super()._transform(X)

        else:
            X_orig = check_X(X)
            X_out = super()._transform(X_orig)
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

    def _get_new_features_name(self) -> List:
        """Return names of the created features."""
        if self.tail == "left":
            indicators = [str(cl) + "_left" for cl in self.variables_]
        elif self.tail == "right":
            indicators = [str(cl) + "_right" for cl in self.variables_]
        else:
            indicators = []
            for cl in self.variables_:
                indicators.append(str(cl) + "_left")
                indicators.append(str(cl) + "_right")
        return indicators

    def _add_new_feature_names(self, feature_names) -> List:
        """Adds names of outlier indicators to transformed variable names."""
        if self.add_indicators is True:
            feature_names = feature_names + self._get_new_features_name()
        return feature_names
