# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import warnings
from typing import Dict, List, Optional, Union

import pandas as pd

from feature_engine._base_transformers.mixins import FitFromDictMixin
from feature_engine._docstrings.fit_attributes import (
    _binner_dict_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.discretisers import (
    _precision_docstring,
    _return_boundaries_docstring,
    _return_object_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _transform_discretiser_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.discretisation.base_discretiser import BaseDiscretiser
from feature_engine.tags import _return_tags


@Substitution(
    return_object=_return_object_docstring,
    return_boundaries=_return_boundaries_docstring,
    precision=_precision_docstring,
    binner_dict_=_binner_dict_docstring,
    transform=_transform_discretiser_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class ArbitraryDiscretiser(BaseDiscretiser, FitFromDictMixin):
    """
    The ArbitraryDiscretiser() divides numerical variables into intervals which limits
    are determined by the user. Thus, it works only with numerical variables.

    You need to enter a dictionary with variable names as keys, and a list with
    the limits of the intervals as values. For example the key could be the variable
    name 'var1' and the value the following list: [0, 10, 100, 1000]. The
    ArbitraryDiscretiser() will then sort var1 values into the intervals 0-10,
    10-100, 100-1000, and var2 into 5-10, 10-15 and 15-20. Similar to `pandas.cut`.

    More details in the :ref:`User Guide <arbitrary_discretiser>`.

    Parameters
    ----------
    binning_dict: dict
        The dictionary with the variable to interval limits pairs.

    {return_object}

    {return_boundaries}

    {precision}

    errors: string, default='ignore'
        Indicates what to do when a value is outside the limits indicated in the
        'binning_dict'. If 'raise', the transformation will raise an error.
        If 'ignore', values outside the limits are returned as NaN
        and a warning will be raised instead.

    Attributes
    ----------
    {binner_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    See Also
    --------
    pandas.cut

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.discretisation import ArbitraryDiscretiser
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.randint(1,100, 100)))
    >>> bins = dict(x =  [0, 25, 50, 75, 100])
    >>> ad = ArbitraryDiscretiser(binning_dict = bins)
    >>> ad.fit(X)
    >>> ad.transform(X)["x"].value_counts()
    2    31
    0    27
    3    25
    1    17
    Name: x, dtype: int64
    """

    def __init__(
        self,
        binning_dict: Dict[Union[str, int], List[Union[str, int]]],
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
        errors: str = "ignore",
    ) -> None:

        if not isinstance(binning_dict, dict):
            raise ValueError(
                "binning_dict must be a dictionary with the interval limits per "
                f"variable. Got {binning_dict} instead."
            )

        if errors not in ["ignore", "raise"]:
            raise ValueError(
                "errors only takes values 'ignore' and 'raise'. "
                f"Got {errors} instead."
            )

        super().__init__(return_object, return_boundaries, precision)

        self.binning_dict = binning_dict
        self.errors = errors

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: None
            y is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = super()._fit_from_dict(X, self.binning_dict)

        # for consistency wit the rest of the discretisers, we add this attribute
        self.binner_dict_ = self.binning_dict

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the variable values into the intervals.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """

        X = super().transform(X)
        # check if NaN values were introduced by the discretisation procedure.
        if X[self.variables_].isnull().sum().sum() > 0:

            # obtain the name(s) of the columns with null values
            nan_columns = (
                X[self.variables_].columns[X[self.variables_].isnull().any()].tolist()
            )

            if len(nan_columns) > 1:
                nan_columns_str = ", ".join(nan_columns)
            else:
                nan_columns_str = nan_columns[0]

            if self.errors == "ignore":
                warnings.warn(
                    f"During the discretisation, NaN values were introduced in "
                    f"the feature(s) {nan_columns_str}."
                )

            elif self.errors == "raise":
                raise ValueError(
                    "During the discretisation, NaN values were introduced in "
                    f"the feature(s) {nan_columns_str}."
                )

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
