# Authors: dodoarg <eargiolas96@gmail.com>

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.datetime.base_transformer import DateTimeBaseTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class ExtractDateFeatures(DateTimeBaseTransformer):
    """
    ExtractDateFeatures extract various date-related features from
    datetime features present in the dataset, adding new columns
    accordingly.

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
        The list of date features to extract. Defaults to extracting all of
        the supported features.
        Note: if you don't specify what features to extract and your dataset
        contains many datetime variables, its feature space might explode!

    drop_datetime: bool, default="True"
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
    supported:
        list of supported date features

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
        Add the new date features specified in features_to_extract argument
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
        drop_datetime: bool = True,
        dayfirst: bool = False,
        yearfirst: bool = False,
        missing_values: str = "raise"
    ) -> None:

        # get the list of supported features from a const variable somewhere?
        self.supported = [
            "month",
            "quarter",
            "semester",
            "year",
            "week_of_the_year",
            "day_of_the_week",
            "day_of_the_month",
            "day_of_the_year",
            "is_weekend",
            "week_of_the_month",
        ]

        if features_to_extract:
            if not isinstance(features_to_extract, list):
                raise TypeError("features_to_extract must be a list of strings")
            elif any(feature not in self.supported for feature in features_to_extract):
                raise ValueError(
                    "At least one of the requested feature is not supported. "
                    "Supported features are {}.".format(", ".join(self.supported))
                )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        self.variables = _check_input_parameter_variables(variables)
        self.drop_datetime = drop_datetime
        self.missing_values = missing_values
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.features_to_extract = features_to_extract

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = super().fit(X, y)
        self.n_features_in_ = X.shape[1]
        if self.features_to_extract is None:
            self.features_to_extract_ = self.supported
        else:
            self.features_to_extract_ = self.features_to_extract
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        for var in self.variables_:
            if "month" in self.features_to_extract_:
                X[str(var) + "_month"] = X[var].dt.month
            if "quarter" in self.features_to_extract_:
                X[str(var) + "_quarter"] = X[var].dt.quarter
            if "semester" in self.features_to_extract_:
                X[str(var) + "_semester"] = np.where(
                    X[var].dt.month <= 6, 1, 2).astype(np.int64)
            if "year" in self.features_to_extract_:
                X[str(var) + "_year"] = X[var].dt.year
            if "week_of_the_year" in self.features_to_extract_:
                X[str(var) + "_woty"] = X[var].dt.isocalendar().\
                    week.astype(np.int64)
            if "day_of_the_week" in self.features_to_extract_:
                X[str(var) + "_dotw"] = X[var].dt.isocalendar().\
                    day.astype(np.int64)
            if "day_of_the_month" in self.features_to_extract_:
                X[str(var) + "_dotm"] = X[var].dt.day
            if "day_of_the_year" in self.features_to_extract_:
                X[str(var) + "_doty"] = X[var].dt.dayofyear
            # maybe add option to choose if friday should be considered a w.e. day?
            if "is_weekend" in self.features_to_extract_:
                X[str(var) + "_is_weekend"] = np.where(
                    X[var].dt.isocalendar().day <= 5, False, True)
            if "week_of_the_month" in self.features_to_extract_:
                X[str(var) + "_wotm"] = X[var].dt.day.apply(
                    lambda d: (d - 1) // 7 + 1)

        if self.drop_datetime:
            X.drop(self.variables_, axis=1, inplace=True)

        return X
