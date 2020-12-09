# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause


from typing import Optional

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe, _check_contains_na
from feature_engine.outliers.base_outlier import BaseOutlier
from feature_engine.parameter_checks import _define_numerical_dict
from feature_engine.variable_manipulation import _find_or_check_numerical_variables


class ArbitraryOutlierCapper(BaseOutlier):
    """
    The ArbitraryOutlierCapper() caps the maximum or minimum values of a variable
    at an arbitrary value indicated by the user.

    The user must provide the maximum or minimum values that will be used
    to cap each variable in a dictionary {feature:capping value}

    Parameters
    ----------
    max_capping_dict : dictionary, default=None
        Dictionary containing the user specified capping values for the right tail of
        the distribution of each variable (maximum values).

    min_capping_dict : dictionary, default=None
        Dictionary containing user specified capping values for the eft tail of the
        distribution of each variable (minimum values).

    missing_values : string, default='raise'
        Indicates if missing values should be ignored or raised. If
        `missing_values='raise'` the transformer will return an error if the
        training or the datasets to transform contain missing values.

    Attributes
    ----------
    right_tail_caps_:
        Dictionary with the maximum values at which variables will be capped.

    left_tail_caps_ :
        Dictionary with the minimum values at which variables will be capped.

    Methods
    -------
    fit:
        This transformer does not learn any parameter.
    transform:
        Cap the variables.
    fit_transform:
        Fit to the data. Then transform it.
    """

    def __init__(
        self,
        max_capping_dict: Optional[dict] = None,
        min_capping_dict: Optional[dict] = None,
        missing_values: str = "raise",
    ) -> None:

        if not max_capping_dict and not min_capping_dict:
            raise ValueError(
                "Please provide at least 1 dictionary with the capping values."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        self.max_capping_dict = _define_numerical_dict(max_capping_dict)
        self.min_capping_dict = _define_numerical_dict(min_capping_dict)
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        self
        """
        X = _is_dataframe(X)

        # find variables to be capped
        if self.min_capping_dict is None and self.max_capping_dict:
            self.variables = [x for x in self.max_capping_dict.keys()]
        elif self.max_capping_dict is None and self.min_capping_dict:
            self.variables = [x for x in self.min_capping_dict.keys()]
        elif self.min_capping_dict and self.max_capping_dict:
            tmp = self.min_capping_dict.copy()
            tmp.update(self.max_capping_dict)
            self.variables = [x for x in tmp.keys()]

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        # find or check for numerical variables
        self.variables = _find_or_check_numerical_variables(X, self.variables)

        if self.max_capping_dict is not None:
            self.right_tail_caps_ = self.max_capping_dict
        else:
            self.right_tail_caps_ = {}

        if self.min_capping_dict is not None:
            self.left_tail_caps_ = self.min_capping_dict
        else:
            self.left_tail_caps_ = {}

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseOutlier.transform.__doc__
