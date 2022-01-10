# Authors: kylegilde <kylegilde@gmail.com>

from typing import List, Optional, Union, cast, Tuple
import itertools

import numpy as np
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

    In other words, DatetimeSubtraction() subtracts a group of features from a group of
    reference variables, and returns the result as new variables in the dataframe.

    The transformed dataframe will contain the additional features indicated in the
    new_variables_name list plus the original set of variables.

    More details in the :ref:`User Guide <datetime_subtraction>`.

    Parameters
    ----------
    variables_to_combine: list
        The list of datetime variables that the reference variables will be subtracted
        from.

        If None, the transformer will find and select all datetime
        variables, including variables of type object that can be converted to datetime.

    reference_variables: list
        The list of datetime reference variables that will be  subtracted from the
         `variables_to_combine`.

         If None, the transformer will find and select all datetime
         variables, including variables of type object that can be converted to datetime

    output_unit: string, default='D'
        The string representation of the output unit of the datetime differences.
        The default is `D` for day. This parameter is passed to numpy.timedelta64.
        Other possible values are  `Y` for year, `M` for month,  `W` for week,
        `h` for hour, `m` for minute, `s` for second, `ms` for millisecond,
        `us` or `μs` for microsecond, `ns` for nanosecond, `ps` for picosecond,
        `fs` for femtosecond and `as` for attosecond.

    dedupe_variable_pairs: bool, default=False
        If `True`, the pairs of variables created from `variables_to_combine` and
        `reference_variables` will be de-duplicated in two ways: variable pairs
        consisting of the same variables and one variable pair if it matches another
        pair when sorted. Setting this to `True` will be useful if you provide any empty
        lists to `variables_to_combine` and `reference_variables` and want to create a
        subtraction feature for every unique pair of datetime variables.

    new_variables_names: list, default=None
        Names of the new variables. If passing a list with the names for the new
        features (recommended), you must enter as many names as new features created
        by the transformer. The number of new features is the number of
        `reference_variables` times the number of `variables_to_combine`.

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
    reference_variables_:
        Contains either the `reference_variables` list that was provided or the
        datetime variables that were found if `reference_variables` was `None`.

    variables_to_combine_:
        Contains either the `variables_to_combine` list that was provided or the
        datetime variables that were found if `variables_to_combine` was `None`.

    variable_pairs_:
        The list of pairs of variables (tuples) that were subtracted.

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
        variables_to_combine: Optional[List[Union[str, int]]] = None,
        reference_variables: Optional[List[Union[str, int]]] = None,
        output_unit: str = 'D',
        dedupe_variable_pairs: bool = False,
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

        valid_output_units = {'D', 'Y', 'M', 'W', 'h', 'm', 's', 'ms', 'us', 'μs', 'ns',
                              'ps', 'fs', 'as'}

        if output_unit not in valid_output_units:
            raise ValueError(f"output_unit accepts the following values: "
                             f"{valid_output_units}")

        if new_variables_names and reference_variables and variables_to_combine:
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

        if not isinstance(dedupe_variable_pairs, bool):
            raise TypeError(
                "dedupe_variable_pairs takes only boolean values True and False. "
                f"Got {dedupe_variable_pairs} instead."
            )

        self.reference_variables = reference_variables
        self.variables_to_combine = variables_to_combine
        self.output_unit = output_unit
        self.dedupe_variable_pairs = dedupe_variable_pairs
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
        self.variables_to_combine_ = _find_or_check_datetime_variables(
            X, self.variables_to_combine
        )

        # find or check for datetime reference variables
        self.reference_variables_ = _find_or_check_datetime_variables(
            X, self.reference_variables
        )

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.reference_variables_)
            _check_contains_na(X, self.variables_to_combine_)

            _check_contains_inf(X, self.reference_variables_)
            _check_contains_inf(X, self.variables_to_combine_)

        variable_pairs = list(itertools.product(self.reference_variables_,
                                                self.variables_to_combine_))

        if self.dedupe_variable_pairs:
            # remove the pairs consisting of the same elements
            # then sort the tuple values and dedupe them
            variable_pairs = \
                cast(
                    List[Tuple[Union[str, int], Union[str, int]]],
                    list({tuple(sorted([var1, var2])) for var1, var2 in variable_pairs
                          if var1 != var2})
                )

        self.variable_pairs_ = variable_pairs

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
            _check_contains_na(X, self.variables_to_combine_)
            _check_contains_na(X, self.variables_to_combine_)

            _check_contains_inf(X, self.variables_to_combine_)
            _check_contains_inf(X, self.variables_to_combine_)

        # convert datetime variables
        datetime_variables = set(self.variables_to_combine_ +
                                 self.variables_to_combine_)

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

        # subtract each pair of variables and convert the values to the output unit
        for var1, var2 in self.variable_pairs_:

            varname = str(var1) + "_sub_" + str(var2) + '_' + self.output_unit

            X[varname] = \
                datetime_df[var1]\
                .sub(datetime_df[var2])\
                .pipe(lambda s: s / np.timedelta64(1, self.output_unit))

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
