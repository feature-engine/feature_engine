# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union
import warnings

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame, IntoSeries

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
class MissingIndicator(BaseImputer):
    """
    The MissingIndicator() adds binary variables that indicate if data is
    missing (one indicator per variable). The added variables (missing indicators) are
    named with the original variable name plus '_na'.

    The MissingIndicator() works for both numerical and categorical variables. You
    can pass a list with the variables for which the missing indicators should be
    added. Alternatively, the imputer will select and add missing indicators to all
    variables in the training set.

    **Note**
    If `missing_only=True`, the imputer will add missing indicators only to those
    variables that show missing data during `fit()`. These may be a subset of the
    variables you indicated in `variables`.

    More details in the :ref:`User Guide <add_missing_indicator>`.

    Parameters
    ----------
    missing_only: bool, default=True
        If missing indicators should be added to variables with missing
        data or to all variables.

        **True**: indicators will be created only for those variables that showed
        missing data during `fit()`.

        **False**: indicators will be created for all variables.

    variables: list, default=None
        The list of variables to impute. If None, the imputer will find and
        select all variables.

    {return_empty}

    Attributes
    ----------
    variables_:
        List of variables for which the missing indicators will be created.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find the variables for which the missing indicators will be created

    {fit_transform}

    transform:
        Add the missing indicators.

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import MissingIndicator
    >>> X = pd.DataFrame(dict(
    >>>        x1 = [np.nan,1,1,0,np.nan],
    >>>        x2 = ["a", np.nan, "b", np.nan, "a"],
    >>>        ))
    >>> ami = MissingIndicator()
    >>> ami.fit(X)
    >>> ami.transform(X)
        x1   x2  x1_na  x2_na
    0  NaN    a      1      0
    1  1.0  NaN      0      1
    2  1.0    b      0      0
    3  0.0  NaN      0      1
    4  NaN    a      1      0

    With polars:

    >>> import polars as pl
    >>> from feature_engine.imputation import MissingIndicator
    >>> X = pl.DataFrame(dict(
    ...        x1 = [None, 1, 1, 0, None],
    ...        x2 = ["a", None, "b", None, "a"],
    ...        ))
    >>> ami = MissingIndicator()
    >>> ami.fit(X)
    >>> ami.transform(X)
    shape: (5, 4)
    ┌──────┬──────┬───────┬───────┐
    │ x1   ┆ x2   ┆ x1_na ┆ x2_na │
    │ ---  ┆ ---  ┆ ---   ┆ ---   │
    │ i64  ┆ str  ┆ i8    ┆ i8    │
    ╞══════╪══════╪═══════╪═══════╡
    │ null ┆ a    ┆ 1     ┆ 0     │
    │ 1    ┆ null ┆ 0     ┆ 1     │
    │ 1    ┆ b    ┆ 0     ┆ 0     │
    │ 0    ┆ null ┆ 0     ┆ 1     │
    │ null ┆ a    ┆ 1     ┆ 0     │
    └──────┴──────┴───────┴───────┘
    """

    def __init__(
        self,
        missing_only: bool = True,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:

        if not isinstance(missing_only, bool):
            raise ValueError("missing_only takes values True or False")

        self.variables = _check_variables_input_value(variables)
        self.missing_only = missing_only

        _check_return_empty_is_bool(return_empty)
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learn the variables for which the missing indicators will be created.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: Series, default=None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        check_X(X)

        # find variables for which indicator should be added
        if self.variables is None:
            variables_ = find_all_variables(X, self.return_empty)
        else:
            variables_ = check_all_variables(X, self.variables)

        if self.missing_only is True:
            # Benchmarked: a per-column isnull().sum() loop is ~2-5x faster
            # than narwhals' single null_count() call on pandas input (the
            # loop calls straight into pandas' C implementation with no
            # narwhals overhead), so pandas keeps its own fast path here.
            is_pandas = nwd.is_pandas_dataframe(X)
            if is_pandas is True:
                variables_ = [
                    var for var in variables_ if X[var].isnull().sum() > 0
                ]
            else:
                nw_X = nw.from_native(X, eager_only=True)
                null_counts = nw_X.select(variables_).null_count().row(0)
                variables_ = [
                    var
                    for var, count in zip(variables_, null_counts)
                    if count > 0
                ]

        self.variables_ = variables_
        self._get_feature_names_in(X)

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Add the binary missing indicators.

        Parameters
        ----------

        X : dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------

        X_new : dataframe of shape = [n_samples, n_features]
            The dataframe containing the additional binary variables.
        """

        X = self._transform(X)

        # Benchmarked: building a separate indicator frame and concatenating
        # it (pandas-native) is ~2-5x faster than narwhals' with_columns
        # equivalent on pandas input, so pandas keeps its own fast path here.
        is_pandas = nwd.is_pandas_dataframe(X)
        if is_pandas is True:
            pd = nw.from_native(X, eager_only=True).__native_namespace__()
            X_indicators = (
                X[self.variables_]
                .isna()
                .astype("int8")
                .add_suffix("_na")
            )
            X = pd.concat([X, X_indicators], axis=1)
        else:
            nw_X = nw.from_native(X, eager_only=True)
            nw_X = nw_X.with_columns(
                nw.col(var).is_null().cast(nw.Int8).alias(f"{var}_na")
                for var in self.variables_
            )
            X = nw_X.to_native()

        return X

    def _get_new_features_name(self) -> List:
        """Return names of the created features."""
        return [f"{feat}_na" for feat in self.variables_]

    def _add_new_feature_names(self, feature_names) -> List:
        """Adds names of new features."""
        return feature_names + self._get_new_features_name()

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


# TODO remove in version 2.1.0

class AddMissingIndicator(MissingIndicator):
    """
    Deprecated alias for MissingIndicator.

    Use MissingIndicator instead.
    """

    def __init__(
        self,
        missing_only: bool = True,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:

        warnings.warn(
            (
                "AddMissingIndicator was deprecated in version 2.0.0 "
                "in favour of MissingIndicator and will be removed in "
                "version 2.1.0. Use MissingIndicator instead."
            ),
            FutureWarning,
            stacklevel=2,
        )

        super().__init__(
            missing_only=missing_only,
            variables=variables,
            return_empty=return_empty,
        )
