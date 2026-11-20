# Authors: dodoarg <eargiolas96@gmail.com>

from typing import List, Optional, Union

import narwhals as nw
import numpy as np
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
from feature_engine.dataframe_helpers import add_columns_to_df, drop_variables_from_df
from feature_engine.datetime._datetime_constants import (
    FEATURES_DEFAULT,
    FEATURES_FUNCTIONS,
    FEATURES_SUFFIXES,
    FEATURES_SUPPORTED,
)
from feature_engine.datetime._datetime_helpers import (
    datetime_to_numpy,
    index_to_datetime_numpy,
)
from feature_engine.variable_handling.check_variables import check_datetime_variables
from feature_engine.variable_handling.find_variables import find_datetime_variables


@Substitution(
    return_empty=_return_empty_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class DatetimeFeatures(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    DatetimeFeatures extracts date and time features from datetime variables, adding
    new columns to the dataset. DatetimeFeatures can extract datetime information from
    existing datetime or object-like variables or from the dataframe index.

    DatetimeFeatures works with pandas and polars dataframes. `dayfirst`,
    `yearfirst` and `utc` are pandas-only parsing options and have no effect on
    polars input, so use `format` there instead.

    The transformer supports the extraction of the following features:

    - "month"
    - "quarter"
    - "semester"
    - "year"
    - "week"
    - "day_of_week"
    - "day_of_month"
    - "day_of_year"
    - "weekend"
    - "month_start"
    - "month_end"
    - "quarter_start"
    - "quarter_end"
    - "year_start"
    - "year_end"
    - "leap_year"
    - "days_in_month"
    - "hour"
    - "minute"
    - "second"

    More details in the :ref:`User Guide <datetime_features>`.

    Parameters
    ----------
    variables: str, list, default=None
        List with the variables from which date and time information will be extracted.
        If None, the transformer will find and select all datetime variables,
        including variables of type object that can be converted to datetime.
        If "index", the transformer will extract datetime features from the
        index of the dataframe. "index" is only supported when `X` is a pandas
        dataframe, since only pandas dataframes have an index.

    {return_empty}

    features_to_extract: list, default=None
        The list of date features to extract. If None, the following features will be
        extracted: "month", "year", "day_of_week", "day_of_month", "hour",
        "minute" and "second". If "all", all supported features will be extracted.
        Alternatively, you can pass a list with the names of the features you want to
        extract.

    drop_original: bool, default="True"
        If True, the original datetime variables will be dropped from the dataframe.

    missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If 'raise' the
        transformer will return an error if the datasets to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when
        performing the feature extraction. Missing data is only evaluated in the
        variables that will be used to derive the date and time features. If features
        are derived from the dataframe index, missing data will be checked in the
        index.

    dayfirst: bool, default="False"
        Specify a date parse order if arg is str or is list-like. If True, parses
        dates with the day first, e.g. 10/11/12 is parsed as 2012-11-10. Same as in
        `pandas.to_datetime`. Only applied when `X` is a pandas dataframe; ignored
        for other backends, which have no equivalent parsing option.

    yearfirst: bool, default="False"
        Specify a date parse order if arg is str or is list-like.
        Same as in `pandas.to_datetime`. Only applied when `X` is a pandas
        dataframe; ignored for other backends, which have no equivalent parsing
        option.

        - If True parses dates with the year first, e.g. 10/11/12 is parsed as
          2010-11-12.
        - If both dayfirst and yearfirst are True, yearfirst takes precedence.

    utc: bool, default=None
        Return UTC DatetimeIndex if True (converting any tz-aware datetime.datetime
        objects as well). Same as in `pandas.to_datetime`. Only applied when `X` is
        a pandas dataframe; ignored for other backends, which have no equivalent
        parsing option.

    format: str, default None
        The strftime to parse time, e.g. "%d/%m/%Y". Check pandas `to_datetime()` for
        more information on choices. If you have variables with different formats pass
        “mixed”, to infer the format for each element individually. This is risky,
        and you should probably use it along with dayfirst, according to pandas'
        documentation.


    Attributes
    ----------
    variables_:
        List of variables from which date and time features will be extracted. If None,
        features will be extracted from the dataframe index.

    features_to_extract_:
        The date and time features that will be extracted from each variable or the
        index.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    transform:
        Add the date and time features.

    See also
    --------
    pandas.to_datetime
    pandas.dt

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.datetime import DatetimeFeatures
    >>> X = pd.DataFrame(dict(date = ["2022-09-18", "2022-10-27", "2022-12-24"]))
    >>> dtf = DatetimeFeatures(features_to_extract = ["year", "month", "day_of_month"])
    >>> dtf.fit(X)
    >>> dtf.transform(X)
        date_year  date_month  date_day_of_month
    0       2022           9                 18
    1       2022          10                 27
    2       2022          12                 24

    With polars:

    >>> import polars as pl
    >>> from feature_engine.datetime import DatetimeFeatures
    >>> X = pl.DataFrame(dict(date = ["2022-09-18", "2022-10-27", "2022-12-24"]))
    >>> dtf = DatetimeFeatures(features_to_extract = ["year", "month", "day_of_month"])
    >>> dtf.fit(X)
    >>> dtf.transform(X)
    shape: (3, 3)
    ┌───────────┬────────────┬───────────────────┐
    │ date_year ┆ date_month ┆ date_day_of_month │
    │ ---       ┆ ---        ┆ ---               │
    │ i64       ┆ i64        ┆ i64               │
    ╞═══════════╪════════════╪═══════════════════╡
    │ 2022      ┆ 9          ┆ 18                │
    │ 2022      ┆ 10         ┆ 27                │
    │ 2022      ┆ 12         ┆ 24                │
    └───────────┴────────────┴───────────────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        features_to_extract: Union[None, str, List[str]] = None,
        drop_original: bool = True,
        missing_values: str = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        utc: Union[None, bool] = None,
        format: Union[None, str] = None,
    ) -> None:

        if features_to_extract:
            if not (
                isinstance(features_to_extract, list) or features_to_extract == "all"
            ):
                raise ValueError(
                    "features_to_extract must be a list of strings or 'all'. "
                    f"Got {features_to_extract} instead."
                )
            elif isinstance(features_to_extract, list) and any(
                feat not in FEATURES_SUPPORTED for feat in features_to_extract
            ):
                raise ValueError(
                    "Some of the requested features are not supported. "
                    "Supported features are {}.".format(", ".join(FEATURES_SUPPORTED))
                )

        if not isinstance(drop_original, bool):
            raise ValueError(
                "drop_original takes only booleans True or False. "
                f"Got {drop_original} instead."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if utc is not None and not isinstance(utc, bool):
            raise ValueError("utc takes only booleans or None. " f"Got {utc} instead.")

        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.drop_original = drop_original
        self.missing_values = missing_values
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.utc = utc
        self.features_to_extract = features_to_extract
        self.format = format

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        This transformer does not learn any parameter.

        Finds datetime variables or checks that the variables selected by the user
        can be converted to datetime.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """
        X = check_X(X)

        if self.variables == "index":
            index = index_to_datetime_numpy(X)
            if self.missing_values == "raise":
                self._check_index_contains_na(index)
            self.variables_ = []

        else:
            if self.variables is None:
                self.variables_ = find_datetime_variables(
                    X, return_empty=self.return_empty
                )
            else:
                self.variables_ = check_datetime_variables(X, self.variables)

            if self.missing_values == "raise":
                _check_contains_na(X, self.variables_)

        if self.features_to_extract is None:
            self.features_to_extract_ = FEATURES_DEFAULT
        elif isinstance(self.features_to_extract, str):
            self.features_to_extract_ = FEATURES_SUPPORTED
        else:
            self.features_to_extract_ = self.features_to_extract

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Extract the date and time features and add them to the dataframe.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe, shape = [n_samples, n_features x n_df_features]
            The dataframe with the original variables plus the new variables.
        """
        check_is_fitted(self)
        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)

        if self.variables == "index":
            index = index_to_datetime_numpy(
                X,
                dayfirst=self.dayfirst,
                yearfirst=self.yearfirst,
                utc=self.utc,
                format=self.format,
            )
            if self.missing_values == "raise":
                self._check_index_contains_na(index)
            new_features = self._extract_features(index)

        else:
            if self.missing_values == "raise":
                _check_contains_na(X, self.variables_)

            if len(self.variables_) == 0:
                return nw.to_native(X, pass_through=True)

            arrays = datetime_to_numpy(
                X,
                self.variables_,
                dayfirst=self.dayfirst,
                yearfirst=self.yearfirst,
                utc=self.utc,
                format=self.format,
                local_time=True,
            )
            new_features = {}
            for var, values in zip(self.variables_, arrays):
                for feat, result in self._extract_features(values).items():
                    new_features[str(var) + FEATURES_SUFFIXES[feat]] = result

        X = add_columns_to_df(X, new_features)
        if self.variables != "index" and self.drop_original is True:
            X = drop_variables_from_df(X, self.variables_)

        return nw.to_native(X, pass_through=True)

    def _extract_features(self, values: np.ndarray) -> dict:
        """Returns the date and time features of one datetime array, by feature."""
        na_mask = np.isnat(values)
        has_na = bool(na_mask.any())
        features = {}
        for feat in self.features_to_extract_:
            result = FEATURES_FUNCTIONS[feat](values)
            if has_na is True:
                result = result.astype("float64")
                result[na_mask] = np.nan
            features[feat] = result
        return features

    def _get_new_features_name(self) -> List:
        """create the names for the datetime features."""

        if self.variables == "index":
            feature_names = [
                FEATURES_SUFFIXES[feat][1:] for feat in self.features_to_extract_
            ]
        else:
            feature_names = [
                str(var) + FEATURES_SUFFIXES[feat]
                for var in self.variables_
                for feat in self.features_to_extract_
            ]

        return feature_names

    def _check_index_contains_na(self, index: np.ndarray) -> None:
        if np.isnat(index).any():
            raise ValueError(
                "The dataframe index contains missing data. "
                "Check and remove those before using this transformer "
                "or set missing_values to False."
            )

    def _more_tags(self):
        tags_dict = {"variables": "datetime"}
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
