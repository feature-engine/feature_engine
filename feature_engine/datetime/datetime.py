# Authors: dodoarg <eargiolas96@gmail.com>

from typing import List, Optional, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.datetime._datetime_constants import (
    FEATURES_DEFAULT,
    FEATURES_FUNCTIONS,
    FEATURES_SUFFIXES,
    FEATURES_SUPPORTED,
)
from feature_engine.docstrings import (
    Substitution,
    _feature_names_in_docstring,
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _n_features_in_docstring,
)
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_datetime_variables,
)


@Substitution(
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """
    DatetimeFeatures extracts date and time features from datetime variables, adding
    new columns to the dataset. DatetimeFeatures is able to extract datetime
    information from existing datetime or object-like variables.

    DatetimeFeatures uses pandas.to_datetime to convert object variables to datetime
    and pandas.dt to extract the features from datetime.

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
    variables: list, default=None
        The list of variables to extract date and time features from.
        If None, the transformer will find and select all datetime variables,
        including variables of type object that can be converted to datetime.
        If "index", the transformer will extract datetime features from the
        index of the dataframe.

    features_to_extract: list, default=None
        The list of date features to extract. If None, the following features will be
        extracted: "month", "year", "day_of_week", "day_of_month", "hour",
        "minute" and "second". If "all", all supported features will be extracted.
        Alternatively, you can pass a list with the names of the supported features
        you want to extract.

    drop_original: bool, default="True"
        If True, the original datetime variables will be dropped from the dataframe.

    missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If 'raise' the
        transformer will return an error if the the datasets to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when
        performing the feature extraction.

    dayfirst: bool, default="False"
        Specify a date parse order if arg is str or its list-likes. If True, parses
        dates with the day first, eg 10/11/12 is parsed as 2012-11-10.

    yearfirst: bool, default="False"
        Specify a date parse order if arg is str or its list-likes.

        - If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
        - If both dayfirst and yearfirst are True, yearfirst is preceded.

    utc: bool, default=None
        Return UTC DatetimeIndex if True (converting any tz-aware datetime.datetime
        objects as well).

    Attributes
    ----------
    variables_:
        List of variables from which date and time features will be extracted.

    features_to_extract_:
        The date and time features that will be extracted from each variable.

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
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        features_to_extract: Union[None, str, List[str]] = None,
        drop_original: bool = True,
        missing_values: str = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        utc: Union[None, bool] = None,
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

        self.variables = _check_input_parameter_variables(variables)
        self.drop_original = drop_original
        self.missing_values = missing_values
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.utc = utc
        self.features_to_extract = features_to_extract

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

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for datetime variables
        self.variables_ = _find_or_check_datetime_variables(X, self.variables)

        # check if datetime variables contains na
        if self.missing_values == "raise":
            if self.variables_ == ["index"]:
                if X.index.isnull().any():
                    raise ValueError(
                        "Some of the variables to transform contain NaN. Check and "
                        "remove those before using this transformer."
                    )
            else:
                _check_contains_na(X, self.variables_)

        if self.features_to_extract is None:
            self.features_to_extract_ = FEATURES_DEFAULT
        elif isinstance(self.features_to_extract, str):
            self.features_to_extract_ = FEATURES_SUPPORTED
        else:
            self.features_to_extract_ = self.features_to_extract

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the date and time features and add them to the dataframe.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features x n_df_features]
            The dataframe with the original variables plus the new variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            if self.variables_ == ["index"]:
                if X.index.isnull().any():
                    raise ValueError(
                        "Some of the variables to transform contain NaN. Check and "
                        "remove those before using this transformer."
                    )

            else:
                _check_contains_na(X, self.variables_)

        if self.variables_ == ["index"]:
            datetime_df = pd.DataFrame(
                pd.to_datetime(
                    X.index,
                    dayfirst=self.dayfirst,
                    yearfirst=self.yearfirst,
                    utc=self.utc,
                ),
                index=X.index,
            )

        else:
            # reorder variables to match train set
            X = X[self.feature_names_in_]

            # convert datetime variables
            datetime_df = pd.concat(
                [
                    pd.to_datetime(
                        X[variable],
                        dayfirst=self.dayfirst,
                        yearfirst=self.yearfirst,
                        utc=self.utc,
                    )
                    for variable in self.variables_
                ],
                axis=1,
            )

        non_dt_columns = datetime_df.columns[~datetime_df.apply(is_datetime)].tolist()
        if non_dt_columns:
            raise ValueError(
                "ValueError: variable(s) "
                + (len(non_dt_columns) * "{} ").format(*non_dt_columns)
                + "could not be converted to datetime. Try setting utc=True"
            )

        # create new features
        if self.variables_ == ["index"]:
            for feat in self.features_to_extract_:
                X["index" + FEATURES_SUFFIXES[feat]] = FEATURES_FUNCTIONS[feat](
                    datetime_df[0]
                )

        else:
            for var in self.variables_:
                for feat in self.features_to_extract_:
                    X[str(var) + FEATURES_SUFFIXES[feat]] = FEATURES_FUNCTIONS[feat](
                        datetime_df[var]
                    )

            if self.drop_original:
                X.drop(self.variables_, axis=1, inplace=True)

        return X

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: list, default=None
            Input features. If `input_features` is `None`, then the names of all the
            variables in the transformed dataset (original + new variables) is returned.
            Alternatively, only the names for the datetime features derived from
            input_features will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        # Create names for all features or just the indicated ones.
        if input_features is None:
            # Create all features.
            input_features_ = self.variables_
        else:
            if not isinstance(input_features, list):
                raise ValueError(
                    f"input_features must be a list. Got {input_features} instead."
                )
            if any([f for f in input_features if f not in self.variables_]):
                raise ValueError(
                    "Some features in input_features were not used to extract new "
                    "variables. You can only get the names of the extracted features "
                    "with this function."
                )
            # Create just indicated features.
            input_features_ = input_features

        # create the names for the lag features
        feature_names = [
            str(var) + FEATURES_SUFFIXES[feat]
            for var in input_features_
            for feat in self.features_to_extract_
        ]

        # Return names of all variables if input_features is None.
        if input_features is None:
            if self.drop_original is True:
                # Remove names of variables to drop.
                original = [
                    f for f in self.feature_names_in_ if f not in self.variables_
                ]
                feature_names = original + feature_names
            else:
                feature_names = self.feature_names_in_ + feature_names

        return feature_names

    def _more_tags(self):
        tags_dict = {"variables": "datetime"}
        return tags_dict
