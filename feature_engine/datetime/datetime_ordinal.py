from typing import List, Optional, Union
import datetime

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
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


@Substitution(
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
    making the ordinal values relative to this date (starting from 1). This can be
    useful for reducing the magnitude of the ordinal values and for aligning them
    to a specific project timeline.

    Parameters
    ----------
    variables: str, list, default=None
        List with the variables from which date and time information will be extracted.
        If None, the transformer will find and select all datetime variables,
        including variables of type object that can be converted to datetime.

    missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If 'raise' the
        transformer will return an error if the datasets to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when
        performing the feature extraction.

    start_date: str, datetime.datetime, default=None
        A reference date from which the ordinal values will be calculated.
        If provided, the ordinal value of `start_date` will be subtracted from
        each datetime variable's ordinal value, and 1 will be added, so the
        `start_date` itself corresponds to an ordinal value of 1.
        If None, the standard `datetime.toordinal()` value will be used.
        The `start_date` can be a string (e.g., "YYYY-MM-DD") or a datetime object.

    drop_original: bool, default=True
        If True, the original datetime variables will be dropped from the dataframe.

    Attributes
    ----------
    variables_:
        List of variables from which date and time features will be extracted.

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
    pandas.to_datetime
    datetime.toordinal

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.datetime import DatetimeOrdinal
    >>> X = pd.DataFrame(dict(date = ["2023-01-01", "2023-01-02", "2023-01-03"]))
    >>> dtf = DatetimeOrdinal(start_date="2023-01-01")
    >>> dtf.fit(X)
    >>> dtf.transform(X)
       date_ordinal
    0             1
    1             2
    2             3
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
        start_date: Union[None, str, datetime.datetime] = None,
        drop_original: bool = True,
    ) -> None:

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if start_date is not None:
            try:
                self.start_date_ = pd.to_datetime(start_date)
            except Exception as e:
                raise ValueError(
                    f"start_date could not be converted to datetime. "
                    f"Got {start_date} instead. Error: {e}"
                )
        else:
            self.start_date_ = None

        if not isinstance(drop_original, bool):
            raise ValueError(
                "drop_original takes only booleans True or False. "
                f"Got {drop_original} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Finds datetime variables or checks that the variables selected by the user
        can be converted to datetime.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series=None
            It is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = check_X(X)

        if self.variables is None:
            self.variables_ = find_datetime_variables(X)
        else:
            self.variables_ = check_datetime_variables(X, self.variables)

        # check if datetime variables contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)

        if self.start_date_ is not None:
            self.start_date_ordinal_ = self.start_date_.toordinal()
        else:
            self.start_date_ordinal_ = None

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the ordinal datetime features and add them to the dataframe.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.
, default
        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features x n_df_features]
            The dataframe with the original variables plus the new variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        # create a copy(to protect original data)
        X_new = X.copy()

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X_new, self.variables_)

        for var in self.variables_:
            # Convert to datetime, then to ordinal
            datetime_series = pd.to_datetime(X_new[var])
            # Handle NaT values: toordinal() raises ValueError for NaT
            ordinal_series = datetime_series.apply(lambda x: x.toordinal() if pd.notna(x) else pd.NA)

            if self.start_date_ordinal_ is not None:
                # Only apply offset if not NaT
                ordinal_series = ordinal_series.apply(lambda x: x - self.start_date_ordinal_ + 1 if pd.notna(x) else pd.NA)

            X_new[str(var) + "_ordinal"] = ordinal_series

        if self.drop_original:
            X_new.drop(self.variables_, axis=1, inplace=True)

        return X_new

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
