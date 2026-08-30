# Authors: Pradumna Suryawanshi <pradumnasuryawanshi@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._base_transformers.mixins import TransformXyMixin
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import check_all_variables, find_all_variables


@Substitution(
    return_empty=_return_empty_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class DropMissingData(BaseImputer, TransformXyMixin):
    """
    DropMissingData() deletes rows containing missing values. It provides
    similar functionality to `pandas.dropna()`, but within the `fit` and `transform`
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

    {return_empty}

    missing_only: bool, default=True
        If `True`, rows will be dropped when they show missing data in variables that
        had missing data during `fit()`. If `False`, rows will be dropped if there is
        missing data in any of the variables. This parameter only works when
        `threshold=None`, otherwise it is ignored.

    threshold: int or float, default=None
        Require that percentage of non-NA values in a row to keep it. If
        `threshold=1`, all variables need to have data to keep the row. If
        `threshold=0.5`, 50% of the variables need to have data to keep the row.
        If `threshold=0.01`, 1% of the variables need to have data to keep the row.
        If `threshold=None`, rows with NA in any of the variables will be dropped.

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

    With polars:

    >>> import polars as pl
    >>> from feature_engine.imputation import DropMissingData
    >>> X = pl.DataFrame(dict(
    ...        x1 = [None, 1, 1, 0, None],
    ...        x2 = ["a", None, "b", None, "a"],
    ...        ))
    >>> dmd = DropMissingData()
    >>> dmd.fit(X)
    >>> dmd.transform(X)
    shape: (1, 2)
    ┌─────┬─────┐
    │ x1  ┆ x2  │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ b   │
    └─────┴─────┘
    """

    def __init__(
        self,
        missing_only: bool = True,
        threshold: Union[None, int, float] = None,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
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

        _check_return_empty_is_bool(return_empty)
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Find the variables for which missing data should be evaluated to decide if a
        row should be dropped.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training data set.

        y: Series or dataframe, default=None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        nw_X = check_X(X)

        # find variables for which indicator should be added
        if self.variables is None:
            variables_ = find_all_variables(X, return_empty=self.return_empty)
        else:
            variables_ = check_all_variables(X, self.variables)

        # If user passes a threshold, then missing_only is ignored:
        if self.threshold is None and self.missing_only is True:
            # Benchmarked: a per-column isnull().sum() loop beats a narwhals-
            # generic call on pandas input, matching MissingIndicator's split.
            if nwd.is_pandas_dataframe(X):
                variables_ = [
                    var for var in variables_ if X[var].isnull().sum() > 0
                ]
            else:
                nw_X = nw.from_native(X, eager_only=True)
                null_counts = nw_X.select(variables_).null_count().row(0)
                variables_ = [
                    var for var, count in zip(variables_, null_counts) if count > 0
                ]

        self.variables_ = variables_
        self._get_feature_names_in(X)

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Remove rows with missing data.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_new: dataframe
            The complete case dataframe for the selected variables, of shape
            [n_samples - n_samples_with_na, n_features]
        """

        X = self._transform(X)
        return self._select_rows(X, keep=True)

    def return_na_data(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Returns the subset of the dataframe with the rows with missing values. That is,
        the subset of the dataframe that would be removed with the `transform()` method.
        This method may be useful in production, for example if we want to store or log
        the removed observations, that is, rows that will not be fed into the model.

        Parameters
        ----------
        X_na: dataframe of shape = [n_samples_with_na, features]
            The subset of the dataframe with the rows with missing data.
        """

        X = self._transform(X)
        return self._select_rows(X, keep=False)

    def _select_rows(self, X: IntoDataFrame, keep: bool) -> IntoDataFrame:
        """
        Shared row-selection logic for transform() (keep=True, rows without
        missing data) and return_na_data() (keep=False, rows with missing
        data). Deriving both from the same "keep" condition, negated for the
        drop side, guarantees the two outputs are always an exact partition
        of X - they can never overlap or leave a row out.
        """
        if len(self.variables_) == 0:
            # dropna(subset=[]) keeps every row: there are no variables to
            # evaluate missingness on, so nothing can ever be "missing".
            if keep is True:
                return X
            is_pandas = nwd.is_pandas_dataframe(X)
            if is_pandas is True:
                return X.iloc[:0]
            return nw.from_native(X, eager_only=True).head(0).to_native()

        is_pandas = nwd.is_pandas_dataframe(X)
        # Benchmarked: a numpy-backed mask beats both pandas' own axis=1
        # isnull()/notna().sum() (a known-slow reduction) and the narwhals
        # path below, so pandas keeps this dedicated fast path.
        if is_pandas is True:
            if self.threshold is not None:
                non_null_count = X[self.variables_].notna().to_numpy().sum(axis=1)
                mask = non_null_count >= len(self.variables_) * self.threshold
            else:
                mask = ~X[self.variables_].isnull().to_numpy().any(axis=1)
            if keep is False:
                mask = ~mask
            return X[mask]
        else:
            nw_X = nw.from_native(X, eager_only=True)
            if self.threshold is not None:
                non_null_count = nw.sum_horizontal(
                    (~nw.col(var).is_null()).cast(nw.Int64)
                    for var in self.variables_
                )
                expr = non_null_count >= len(self.variables_) * self.threshold
            else:
                expr = ~nw.any_horizontal(
                    (nw.col(var).is_null() for var in self.variables_),
                    ignore_nulls=True,
                )
            if keep is False:
                expr = ~expr
            return nw_X.filter(expr).to_native()

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
