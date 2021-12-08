# Authors: dodoarg <eargiolas96@gmail.com>

from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_datetime_variables,
)

from feature_engine.datetime.datetime_constants import (
    FEATURES_DEFAULT,
    FEATURES_SUPPORTED,
    FEATURES_SUFFIXES,
    FEATURES_FUNCTIONS,
)


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """
    ExtractDatetimeFeatures extract various date and time features from
    datetime variables present in the dataset, adding new columns
    accordingly. The transformer is able to extract datetime information
    present in object-like variables.

    **Notes**
    Day of the month extraction isn't supported by panda modules.
    This transformer implements a workaround that counts how many blocks
    of 7 days have passed in a given month.
    A more serious approach based on the calendar should be implemented
    ***

    Parameters
    ----------
    variables: list, default=None
        The list of variables to extract date features from.
        If None, the imputer will find and select all datetime variables,
        including those converted from object-like features.

    features_to_extract: list, default=None
        The list of date features to extract. Defaults to month, year, of

        Note: if you don't specify what features to extract and your dataset
        contains many datetime variables, its feature space might explode!

    drop_original: bool, default="True"
        Whether to drop datetime variables from the dataframe, including
        those that have been converted by the transformer.
        Note: if you pass a subset of features via the variables argument
        of the transformer and drop_datetime=True, there might be more
        datetime features in the dataframe that will not be dropped
        upon calling the transform method.

    dayfirst: bool, default="False"
        Specify a date parse order for object-like variables. If True,
        parses the date with the day first.

    yearfirst: bool, default="False"
        Specify a date parse order for object-like variables. If True,
        parses the date with the year first.

    Attributes
    ----------

    Available upon calling the fit method:

    variables_:
        List of variables from which to extract date features

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the variables from which to extract date features
    transform:
        Add the new datetime features specified in features_to_extract argument
    fit_transform:
        Fit to the data, then transform it.

    See also
    --------
    pandas.to_datetime
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        features_to_extract: List[str] = None,
        drop_original: bool = True,
        dayfirst: bool = False,
        yearfirst: bool = False,
        missing_values: str = "raise",
    ) -> None:

        if features_to_extract:
            if (
                not isinstance(features_to_extract, list)
                or features_to_extract != "all"
            ):
                raise ValueError(
                    "features_to_extract must be a list of strings or 'all'. "
                    f"Got {features_to_extract} instead."
                )
            elif any(feat not in FEATURES_SUPPORTED for feat in features_to_extract):
                raise ValueError(
                    "Some of the requested features are not supported. "
                    "Supported features are {}.".format(", ".join(FEATURES_SUPPORTED))
                )

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

        self.variables = _check_input_parameter_variables(variables)
        self.drop_original = drop_original
        self.missing_values = missing_values
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.features_to_extract = features_to_extract

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for datetime variables
        self.variables_ = _find_or_check_datetime_variables(X, self.variables)

        # check if datetime variables contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)

        if self.features_to_extract is None:
            self.features_to_extract_ = FEATURES_DEFAULT
        elif self.features_to_extract == "all":
            self.features_to_extract_ = FEATURES_SUPPORTED
        else:
            self.features_to_extract_ = self.features_to_extract

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)

        # convert datetime variables
        datetime_df = pd.concat(
            [
                pd.to_datetime(
                    X[variable], dayfirst=self.dayfirst, yearfirst=self.yearfirst
                )
                for variable in self.variables_
            ],
            axis=1,
        )

        # create new features
        for var in self.variables_:
            for feat in self.features_to_extract_:
                X[str(var) + FEATURES_SUFFIXES[feat]] = FEATURES_FUNCTIONS[feat](
                    datetime_df[var]
                )

        if self.drop_original:
            X.drop(self.variables_, axis=1, inplace=True)

        return X
