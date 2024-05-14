# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import pandas as pd

from feature_engine._base_transformers.mixins import TransformXyMixin
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
class OutlierTrimmer(WinsorizerBase, TransformXyMixin):
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

    fold_:
        Factor multiplying the std, mad, iqr or alternative the percentile. Only
        different from `fold` when `fold="auto"`.

    Methods
    -------
    fit:
        Find maximum and minimum values.

    {fit_transform}

    transform:
        Remove outliers.

    transform_x_y:
        Remove rows with outliers from X set and y.

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

    >>> import pandas as pd
    >>> from feature_engine.outliers import OutlierTrimmer
    >>> X = pd.DataFrame(dict(x = [0.49671,
    >>>                         -0.1382,
    >>>                          0.64768,
    >>>                          1.52302,
    >>>                         -0.2341,
    >>>                         -17.2341,
    >>>                          1.57921,
    >>>                          0.76743,
    >>>                         -0.4694,
    >>>                          0.54256]))
    >>> ot = OutlierTrimmer(capping_method='gaussian', tail='left', fold=3)
    >>> ot.fit(X)
    >>> ot.transform(X)
              x
    0   0.49671
    1  -0.13820
    2   0.64768
    3   1.52302
    4  -0.23410
    5 -17.23410
    6   1.57921
    7   0.76743
    8  -0.46940
    9   0.54256

    >>> import pandas as pd
    >>> from feature_engine.outliers import OutlierTrimmer
    >>> X = pd.DataFrame(dict(x = [0.49671,
    >>>                         -0.1382,
    >>>                          0.64768,
    >>>                          1.52302,
    >>>                         -0.2341,
    >>>                         -17.2341,
    >>>                          1.57921,
    >>>                          0.76743,
    >>>                         -0.4694,
    >>>                          0.54256]))
    >>> ot = OutlierTrimmer(capping_method='mad', tail='left', fold=3)
    >>> ot.fit(X)
    >>> ot.transform(X)
             x
    0  0.49671
    1 -0.13820
    2  0.64768
    3  1.52302
    4 -0.23410
    6  1.57921
    7  0.76743
    8 -0.46940
    9  0.54256
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
            inliers = X[feature].le(self.right_tail_caps_[feature])
            X = X.loc[inliers]

        for feature in self.left_tail_caps_.keys():
            inliers = X[feature].ge(self.left_tail_caps_[feature])
            X = X.loc[inliers]

        return X
