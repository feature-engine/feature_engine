# Authors: Pradumna Suryawanshi <pradumnasuryawanshi@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._base_transformers.mixins import TransformXyMixin
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import check_all_variables, find_all_variables


@Substitution(
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class DropMissingData(BaseImputer, TransformXyMixin):
    """
    DropMissingData() deletes rows containing missing values. It provides
    similar functionality to `pandas.drop_na()`, but within the `fit` and `transform`
    framework.

    It works for numerical and categorical variables. You can enter the list of
    variables for which missing values should be removed. Alternatively, the imputer
    will find and remove missing data in all dataframe variables.

    More details in the :ref:`User Guide <drop_missing_data>`.

    Parameters
    ----------
    variables: list, default=None
        The list of variables to consider for the imputation. If `None`, the imputer
        will check missing data in all variables in the dataframe. Alternatively, the
        imputer will evaluate missing data only in the variables in the list.

        Note that if `missing_only=True`, missing data will be removed from variables
        that had missing data in the train set. These might be a subset of the
        variables indicated in the list.

    missing_only: bool, default=True
        If `True`, rows will be dropped when they show missing data in variables that
        had missing data during `fit()`. If `False`, rows will be dropped if there is
        missing data in any of the variables. This parameter only works when
        `threshold=None`, otherwise it is ignored.

    threshold: int or float, default=None
        Require that percentage of non-NA values in a row to keep it. If
        `threshold=1`, all variables need to have data to keep the row. If
        `threshold=0.5`, 50% of the variables need to have data to keep the row.
        If `threshold=0.01`, 10% of the variables need to have data to keep the row.
        If `thresh=None`, rows with NA in any of the variables will be dropped.

    Attributes
    ----------
    variables_:
        The variables for which missing data will be examined to decide if a row is
        dropped. The attribute `variables_` is different from the parameter `variables`
        when the latter is `None`, or when only a subset of the indicated variables
        show NA in the train set if `missing_only=True`.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find the variables for which missing data should be evaluated.

    {fit_transform}

    return_na_data:
        Returns a dataframe with the rows that contain missing data.

    transform:
        Remove rows with missing data.

    transform_x_y:
        Remove rows with missing data from X and y.

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import DropMissingData
    >>> X = pd.DataFrame(dict(
    >>>        x1 = [np.nan,1,1,0,np.nan],
    >>>        x2 = ["a", np.nan, "b", np.nan, "a"],
    >>>        ))
    >>> dmd = DropMissingData()
    >>> dmd.fit(X)
    >>> dmd.transform(X)
        x1 x2
    2  1.0  b
    """

    def __init__(
        self,
        missing_only: bool = True,
        threshold: Union[None, int, float] = None,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if not isinstance(missing_only, bool):
            raise ValueError(
                "missing_only takes values True or False. "
                f"Got {missing_only} instead."
            )

        if threshold is not None:
            if not isinstance(threshold, (int, float)) or not (0 < threshold <= 1):
                raise ValueError(
                    "threshold must be a value between 0 < x <= 1. "
                    f"Got {threshold} instead."
                )

        self.variables = _check_variables_input_value(variables)
        self.missing_only = missing_only
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Find the variables for which missing data should be evaluated to decide if a
        row should be dropped.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training data set.

        y: pandas Series or dataframe, default=None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        X = check_X(X)

        # find variables for which indicator should be added
        if self.variables is None:
            self.variables_ = find_all_variables(X)
        else:
            self.variables_ = check_all_variables(X, self.variables)

        # If user passes a threshold, then missing_only is ignored:
        if self.threshold is None and self.missing_only is True:
            self.variables_ = [
                var for var in self.variables_ if X[var].isnull().sum() > 0
            ]

        self._get_feature_names_in(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing data.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_new: pandas dataframe
            The complete case dataframe for the selected variables, of shape
            [n_samples - n_samples_with_na, n_features]
        """

        X = self._transform(X)

        if self.threshold:
            X.dropna(
                thresh=len(self.variables_) * self.threshold,
                subset=self.variables_,
                axis=0,
                inplace=True,
            )
        else:
            X.dropna(axis=0, how="any", subset=self.variables_, inplace=True)

        return X

    def return_na_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the subset of the dataframe with the rows with missing values. That is,
        the subset of the dataframe that would be removed with the `transform()` method.
        This method may be useful in production, for example if we want to store or log
        the removed observations, that is, rows that will not be fed into the model.

        Parameters
        ----------
        X_na: pandas dataframe of shape = [n_samples_with_na, features]
            The subset of the dataframe with the rows with missing data.
        """

        X = self._transform(X)

        if self.threshold:
            idx = pd.isnull(X[self.variables_]).mean(axis=1) >= self.threshold
            idx = idx[idx]
        else:
            idx = pd.isnull(X[self.variables_]).any(axis=1)
            idx = idx[idx]

        return X.loc[idx.index, :]

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
