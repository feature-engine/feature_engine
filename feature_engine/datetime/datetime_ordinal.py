import datetime
from typing import List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from dateutil.parser import parse as _parse_datetime
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.variable_handling.check_variables import check_datetime_variables
from feature_engine.variable_handling.find_variables import find_datetime_variables

# datetime.date(1970, 1, 1).toordinal() - the proleptic Gregorian ordinal of the
# Unix epoch, used to convert epoch-based timestamps into the same "days since
# January 1, 0001" ordinal that datetime.date.toordinal() returns.
_UNIX_EPOCH_ORDINAL = 719_163
_MICROSECONDS_PER_DAY = 86_400_000_000


@Substitution(
    return_empty=_return_empty_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class DatetimeOrdinal(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    DatetimeOrdinal transforms datetime variables into their ordinal representation.
    The ordinal representation is an integer value representing the number of days
    since January 1, 0001 in the Gregorian calendar.

    Optionally, a `start_date` can be provided to set a custom reference point,
    making the ordinal values relative to this date (starting from 1).

    More details in the :ref:`User Guide <datetime_ordinal>`.

    Parameters
    ----------
    variables: str, list, default=None
        List of the variables to convert into ordinal. If None, the transformer will
        find and select all datetime variables, including variables of type object that
        can be converted to datetime.

    {return_empty}

    missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If 'raise' the
        transformer will return an error if the datasets passed to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when
        performing the transformation.

    start_date: str, datetime.date, datetime.datetime, default=None
        A reference date from which the ordinal values will be calculated.
        If provided, the ordinal value of `start_date` will be 1, the day after will be
        2, and so on. Days before `start_date` will take negative values.
        If None, the transformation will represent the number of days since
        January 1, 0001. `start_date` can be a string (e.g., "YYYY-MM-DD")
        or a datetime object.

    drop_original: bool, default=True
        If True, the original datetime variables will be dropped from the dataframe
        after the transformation.

    Attributes
    ----------
    variables_:
        List of variables to convert into ordinals.

    start_date_ordinal_:
        The ordinal value of the provided `start_date`, if applicable.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    transform:
        Add the ordinal datetime features.

    See also
    --------
    feature_engine.datetime.DatetimeFeatures
    feature_engine.datetime.DatetimeSubtraction

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.datetime import DatetimeOrdinal
    >>> X = pd.DataFrame(dict(date = ["2023-01-01", "2023-01-02", "2023-01-03"]))
    >>> dtf = DatetimeOrdinal(start_date="2023-01-01")
    >>> _ = dtf.fit(X)
    >>> dtf.transform(X)
       date_ordinal
    0             1
    1             2
    2             3

    With polars:

    >>> import polars as pl
    >>> from feature_engine.datetime import DatetimeOrdinal
    >>> X = pl.DataFrame(dict(date = ["2023-01-01", "2023-01-02", "2023-01-03"]))
    >>> dtf = DatetimeOrdinal(start_date="2023-01-01")
    >>> _ = dtf.fit(X)
    >>> dtf.transform(X)
    shape: (3, 1)
    ┌──────────────┐
    │ date_ordinal │
    │ ---          │
    │ i64          │
    ╞══════════════╡
    │ 1            │
    │ 2            │
    │ 3            │
    └──────────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        missing_values: str = "raise",
        start_date: Union[None, str, datetime.datetime] = None,
        drop_original: bool = True,
    ) -> None:

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if not isinstance(drop_original, bool):
            raise ValueError(
                "drop_original takes only booleans True or False. "
                f"Got {drop_original} instead."
            )

        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.missing_values = missing_values
        self.start_date = start_date
        self.drop_original = drop_original

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        This transformer does not learn any parameter.

        Finds datetime variables or checks that the variables selected by the user
        can be converted to datetime. Also parses `start_date`, if provided, into
        its ordinal representation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: Series=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        ValueError
            If `start_date` was provided but cannot be parsed into a date.
        """
        # check input dataframe
        X = check_X(X)

        # parse the user-provided start_date into its ordinal representation.
        # datetime.datetime is a subclass of datetime.date, so both are handled
        # by the isinstance branch; strings are parsed with dateutil.
        self.start_date_ordinal_: Optional[int]
        if self.start_date is None:
            self.start_date_ordinal_ = None
        elif isinstance(self.start_date, datetime.date):
            self.start_date_ordinal_ = self.start_date.toordinal()
        else:
            try:
                self.start_date_ordinal_ = _parse_datetime(self.start_date).toordinal()
            except Exception as e:
                raise ValueError(
                    f"start_date could not be converted to datetime. "
                    f"Got {self.start_date} instead. Error: {e}"
                )

        if self.variables is None:
            self.variables_ = find_datetime_variables(
                X, return_empty=self.return_empty
            )
        else:
            self.variables_ = check_datetime_variables(X, self.variables)

        # check if datetime variables contains na
        # nw.col([]) errors on the polars backend, so skip when there's
        # nothing to check (happens when return_empty=True found no variables).
        if self.missing_values == "raise" and len(self.variables_) > 0:
            _check_contains_na(X, self.variables_)

        # save input features
        if nwd.is_pandas_dataframe(X):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = nw.from_native(X, eager_only=True).columns

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Calculate ordinal representation of datetime features and add them to the
        dataframe.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe, shape = [n_samples, n_features x n_df_features]
            The dataframe with the original variables plus the new features.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        if len(self.variables_) == 0:
            return X

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)

        nw_X = nw.from_native(X, eager_only=True)

        # variables can be native Date/Datetime columns, or string/categorical
        # columns holding parseable date values - the latter need parsing into
        # a real datetime dtype before the ordinal can be computed.
        schema = nw_X.schema
        to_parse = [
            var
            for var in self.variables_
            if not isinstance(schema[var], (nw.Date, nw.Datetime))
        ]
        if len(to_parse) > 0:
            nw_X = nw_X.with_columns(
                nw.col(var).cast(nw.String).str.to_datetime() for var in to_parse
            )

        if nwd.is_pandas_dataframe(X):
            return self._transform_pandas(nw_X.to_native())
        return self._transform_narwhals(nw_X)

    def _transform_pandas(self, X):
        """Vectorized ordinal computation via numpy datetime64[D] arithmetic.

        Benchmarked ~3.5-12x faster than the narwhals-generic dt.timestamp path
        at 10k-100k rows x 1-10 columns (the gap widens with more columns), so
        pandas keeps its own numpy fast path here.
        """
        new_columns = {}
        for var in self.variables_:
            days = X[var].to_numpy().astype("datetime64[D]")
            na_mask = np.isnat(days)
            ordinal = days.astype("int64") + _UNIX_EPOCH_ORDINAL
            if self.start_date_ordinal_ is not None:
                ordinal = ordinal - self.start_date_ordinal_ + 1
            if na_mask.any():
                # int64 arithmetic on the NaT sentinel can wrap around, but that's
                # harmless - the masked slots are overwritten with NaN right after.
                ordinal = ordinal.astype("float64")
                ordinal[na_mask] = np.nan
            new_columns[str(var) + "_ordinal"] = ordinal

        # assign() still inserts columns one at a time internally, so it doesn't
        # avoid fragmentation with many variables; building one DataFrame and
        # joining it does (single insertion), same pattern as DecisionTreeFeatures.
        X = X.join(type(X)(new_columns, index=X.index))
        if self.drop_original is True:
            X = X.drop(columns=self.variables_)
        return X

    def _transform_narwhals(self, nw_X):
        """Ordinal computation via narwhals' dt.timestamp, already vectorized and
        fast enough on polars that a numpy round-trip wouldn't pay for itself."""
        exprs = []
        for var in self.variables_:
            ordinal_expr = (
                nw.col(var).dt.timestamp("us") // _MICROSECONDS_PER_DAY
                + _UNIX_EPOCH_ORDINAL
            )
            if self.start_date_ordinal_ is not None:
                ordinal_expr = ordinal_expr - self.start_date_ordinal_ + 1
            exprs.append(ordinal_expr.alias(str(var) + "_ordinal"))

        nw_X = nw_X.with_columns(*exprs)
        if self.drop_original is True:
            nw_X = nw_X.drop(self.variables_)
        return nw_X.to_native()

    def _get_new_features_name(self) -> List:
        """create the names for the new features."""
        feature_names = [str(var) + "_ordinal" for var in self.variables_]
        return feature_names

    def _more_tags(self):
        tags_dict = {"variables": "datetime"}
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
