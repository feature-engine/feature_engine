# Authors: kylegilde <kylegilde@gmail.com>

from typing import List, Optional, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _find_or_check_datetime_variables,
)


class DatetimeSubtraction(BaseEstimator, TransformerMixin):
    """
    DatetimeSubtraction() applies datetime subtraction between a group
    of variables and one or more reference features. It adds one or more additional
    features to the dataframe with the result of the operations.

    In other words, DatetimeSubtraction() subtracts a group of features from a group of reference variables, and returns the
    result as new variables in the dataframe.

    The transformed dataframe will contain the additional features indicated in the
    new_variables_name list plus the original set of variables.

    More details in the :ref:`User Guide <datetime_subtraction>`.

    Parameters
    ----------
    variables_to_combine: list
        The list of datetime variables that the reference variables will be subtracted from.

        If None, the transformer will find and select all datetime variables,
        including variables of type object that can be converted to datetime.

    reference_variables: list
        The list of datetime reference variables that will be  subtracted from the
         `variables_to_combine`.

         If None, the transformer will find and select all datetime
         variables, including variables of type object that can be converted to datetime.

    output_unit: string, default='days'
        Indicates the units to use for the difference values. Possible output units include
        `days`, `microseconds`, `nanoseconds` and `seconds`.

    new_variables_names: list, default=None
        Names of the new variables. If passing a list with the names for the new
        features (recommended), you must enter as many names as new features created
        by the transformer. The number of new features is the number of `reference_variables`
        times the number of `variables_to_combine`.

        If `new_variable_names` is None, the transformer will assign an arbitrary name
        to the features. The name will be var + `_sub_` + ref_var.

    missing_values: string, default='ignore'
        Indicates if missing values should be ignored or raised. If 'ignore', the
        transformer will ignore missing data when transforming the data. If 'raise' the
        transformer will return an error if the training or the datasets to transform
        contain missing values.

    drop_original: bool, default=False
        If True, the original variables will be dropped from the dataframe
        after their combination.

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
    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        This transformer does not learn parameters.
    transform:
        Combine the variables with the mathematical operations.
    fit_transform:
        Fit to the data, then transform it.

    """

    def __init__(
        self,
        variables_to_combine: List[Union[str, int]] = None,
        reference_variables: List[Union[str, int]] = None,
        output_unit: str = 'days',
        new_variables_names: Optional[List[str]] = None,
        missing_values: str = "ignore",
        drop_original: bool = False,
        dayfirst: bool = False,
        yearfirst: bool = False,
        utc: Union[None, bool] = None,
    ) -> None:

        # check input types
        if variables_to_combine:
            if not isinstance(variables_to_combine, list) or not all(
                isinstance(var, (int, str)) for var in variables_to_combine
            ):
                raise ValueError(
                    "variables_to_combine takes a list of strings or integers "
                    "corresponding to the names of the variables to be used as  "
                    "reference to combine with the binary operations."
                )

        if reference_variables:
            if not isinstance(variables_to_combine, list) or not all(
                isinstance(var, (int, str)) for var in variables_to_combine
            ):
                raise ValueError(
                    "variables_to_combine takes a list of strings or integers "
                    "corresponding to the names of the variables to combine "
                    "with the binary operations."
                )

        if output_unit not in ['days', 'microseconds', 'nanoseconds', 'seconds']:
            raise ValueError("output_unit takes only values 'days', 'microseconds', "
                             "'nanoseconds', or 'seconds'")

        if new_variables_names:
            if len(new_variables_names) != (
                len(reference_variables) * len(variables_to_combine)
            ):
                raise ValueError(
                    "Number of items in new_variables_names must be equal to number of "
                    "items in reference_variables * items in variables to "
                    "combine. In other words, "
                    "the transformer needs as many new variable names as reference "
                    "variables to perform over the variables to "
                    "combine."
                )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        if not isinstance(drop_original, bool):
            raise TypeError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )

        self.reference_variables = reference_variables
        self.variables_to_combine = variables_to_combine
        self.output_unit = output_unit
        self.new_variables_names = new_variables_names
        self.missing_values = missing_values
        self.drop_original = drop_original
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.utc = utc

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, or np.array. Default=None.
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for datetime variables to combine
        self.variables_to_combine = _find_or_check_datetime_variables(
            X, self.variables_to_combine
        )

        # find or check for datetime reference variables
        self.reference_variables = _find_or_check_datetime_variables(
            X, self.reference_variables
        )

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.reference_variables)
            _check_contains_na(X, self.variables_to_combine)

            _check_contains_inf(X, self.reference_variables)
            _check_contains_inf(X, self.variables_to_combine)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Subtract the groups of datetime variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features + n_operations]
            The dataframe with the new variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.reference_variables)
            _check_contains_na(X, self.variables_to_combine)

            _check_contains_inf(X, self.reference_variables)
            _check_contains_inf(X, self.variables_to_combine)

        # convert datetime variables
        datetime_variables = set(self.variables_to_combine + self.reference_variables)

        datetime_df = X[datetime_variables].apply(pd.to_datetime,
                                                  dayfirst=self.dayfirst,
                                                  yearfirst=self.yearfirst,
                                                  utc=self.utc
                                                  )

        # find any non-datetime columns
        non_dt_columns = datetime_df.apply(is_datetime).loc[lambda x: ~x].index.tolist()
        if non_dt_columns:
            raise ValueError(
                "ValueError: variable(s) " +
                (len(non_dt_columns) * '{} ').format(*non_dt_columns) +
                "could not be converted to datetime. Try setting utc=True"
            )

        original_col_names = X.columns.tolist()

        # Add new features and values into the DataFrame. Set the output units.
        for reference in self.reference_variables:
            varname = [
                str(var) + "_sub_" + str(reference)
                for var in self.variables_to_combine
            ]
            X[varname] = \
                datetime_df[self.variables_to_combine]\
                .sub(datetime_df[reference], axis=0)\
                .apply(lambda col: getattr(col.dt, self.output_unit))

        # replace created variable names with user ones.
        if self.new_variables_names:
            X.columns = original_col_names + self.new_variables_names

        # drop the datetime variables.
        if self.drop_original:
            X.drop(columns=datetime_variables, inplace=True)

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"

        msg = "this transformer works with datasets that contain at least 2 variables. \
        Otherwise, there is nothing to combine"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg
        return tags_dict
