from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.utils.validation import check_is_fitted

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _transform_creation_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.creation.base_creation import BaseCreation
from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.variable_handling.check_variables import check_datetime_variables
from feature_engine.variable_handling.find_variables import find_datetime_variables

_example = """
    >>> import pandas as pd
    >>> from feature_engine.datetime import DatetimeSubtraction
    >>> X = pd.DataFrame({
    >>>     "date1": ["2022-09-18", "2022-10-27", "2022-12-24"],
    >>>     "date2": ["2022-08-18", "2022-08-27", "2022-06-24"]})
    >>> dtf = DatetimeSubtraction(variables=["date1"], reference=["date2"])
    >>> dtf.fit(X)
    >>> dtf.transform(X)
            date1       date2  date1_sub_date2
    0  2022-09-18  2022-08-18             31.0
    1  2022-10-27  2022-08-27             61.0
    2  2022-12-24  2022-06-24            183.0
        """.rstrip()


@Substitution(
    missing_values=_missing_values_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    transform=_transform_creation_docstring,
    fit_transform=_fit_transform_docstring,
    example=_example,
)
class DatetimeSubtraction(BaseCreation):
    """
    DatetimeSubtraction() applies datetime subtraction between a group of datetime
    variables and one or more datetime features, adding the resulting variables to the
    dataframe.

    DatetimeSubtraction() works with variables cast as datetime or object. It subtracts
    the variables listed in the parameter `reference` from those listed in the
    parameter `variables`.

    More details in the :ref:`User Guide <datetime_subtraction>`.

    Parameters
    ----------
    variables: list
        The list of datetime variables that the reference variables will be subtracted
        from (left side of the subtraction operation).

    reference: list
        The list of datetime reference variables that will be subtracted from
        `variables` (right side of the subtraction operation).

    new_variables_names: list, default=None
        Names of the new variables. You have the option to pass a list with the names
        you'd like to assing to the new variables. If `None`, the transformer will
        assign arbitrary names.

    output_unit: string, default='D'
        The string representation of the output unit of the datetime differences.
        The default is `D` for day. This parameter is passed to `numpy.timedelta64`.
        Other possible values are  `Y` for year, `M` for month,  `W` for week,
        `h` for hour, `m` for minute, `s` for second, `ms` for millisecond,
        `us` or `μs` for microsecond, `ns` for nanosecond, `ps` for picosecond,
        `fs` for femtosecond and `as` for attosecond.

    {missing_values}

    drop_original: bool, default="False"
        If `True`, the variables listed in `variables` and `reference` will be dropped
        from the dataframe after the computation of the new features.

    dayfirst: bool, default="False"
        Specify a date parse order if arg is str or is list-like. If True, parses
        dates with the day first, e.g. 10/11/12 is parsed as 2012-11-10. Same as in
        `pandas.to_datetime`.

    yearfirst: bool, default="False"
        Specify a date parse order if arg is str or is list-like.
        Same as in `pandas.to_datetime`.

        - If True parses dates with the year first, e.g. 10/11/12 is parsed as
          2010-11-12.
        - If both dayfirst and yearfirst are True, yearfirst is preceded.

    utc: bool, default=None
        Return UTC DatetimeIndex if True (converting any tz-aware datetime.datetime
        objects as well). Same as in `pandas.to_datetime`.

    format: str, default None
        The strftime to parse time, e.g. "%d/%m/%Y". Check pandas `to_datetime()` for
        more information on choices. If you have variables with different formats pass
        “mixed”, to infer the format for each element individually. This is risky,
        and you should probably use it along with dayfirst, according to pandas'
        documentation.

    Attributes
    ----------
    variables_:
        The list with datetime variables from which the variables in `reference` will
        be substracted. It is created after the transformer corroborates that the
        variables in `variables` are, or can be parsed to datetime.

    reference_:
        The list with the datetime variables that will be subtracted from `variables_`.
        It is created after the transformer corroborates that the variables in
        `reference` are, or can be parsed to datetime.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    Examples
    --------

    {example}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        reference: Union[None, int, str, List[Union[str, int]]] = None,
        new_variables_names: Union[None, List[str], str] = None,
        output_unit: str = "D",
        missing_values: str = "ignore",
        drop_original: bool = False,
        dayfirst: bool = False,
        yearfirst: bool = False,
        utc: Union[None, bool] = None,
        format: Union[None, str] = None,
    ) -> None:

        valid_output_units = {
            "D",
            "Y",
            "M",
            "W",
            "h",
            "m",
            "s",
            "ms",
            "us",
            "μs",
            "ns",
            "ps",
            "fs",
            "as",
        }

        if not isinstance(output_unit, str) or output_unit not in valid_output_units:
            raise ValueError(
                f"output_unit accepts the following values: "
                f"{valid_output_units}. Got {output_unit} instead."
            )

        if new_variables_names is not None:
            if (
                not isinstance(new_variables_names, list)
                or not all(isinstance(var, str) for var in new_variables_names)
                or len(set(new_variables_names)) != len(new_variables_names)
            ):
                raise ValueError(
                    "new_variable_names should be None or a list of unique strings. "
                    f"Got {new_variables_names} instead."
                )

        super().__init__(missing_values, drop_original)
        self.variables = _check_variables_input_value(variables)
        self.reference = _check_variables_input_value(reference)
        self.new_variables_names = new_variables_names
        self.output_unit = output_unit
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.utc = utc
        self.format = format

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
        # Common checks and attributes
        X = check_X(X)

        # check variables are datetime
        if self.variables is None:
            self.variables_ = find_datetime_variables(X)
        else:
            self.variables_ = check_datetime_variables(X, self.variables)

        if self.reference is None:
            self.reference_ = find_datetime_variables(X)
        else:
            self.reference_ = check_datetime_variables(X, self.reference)

        if self.new_variables_names is not None:
            if len(self.new_variables_names) != len(self.variables_) * len(
                self.reference_
            ):
                raise ValueError(
                    f"{len(self.variables_) * len(self.reference_)} new variables will "
                    f"be created but only {len(self.new_variables_names)} new variable "
                    f"names were provided. Please check the variables list and try "
                    f"again."
                )

        # check if dataset contains na
        if self.missing_values == "raise":
            vars = list(set(self.variables_ + self.reference_))
            _check_contains_na(X, vars)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add new features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe
            The input dataframe plus the new variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        if self.missing_values == "raise":
            vars = list(set(self.variables_ + self.reference_))
            _check_contains_na(X, vars)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        X_dt = self._to_datetime(X)

        new_features = self._sub(X_dt)

        X = pd.concat([X, new_features], axis=1)

        if self.drop_original:
            X = X.drop(
                columns=set(self.variables_ + self.reference_),
            )

        return X

    def _to_datetime(self, X: pd.DataFrame):
        """covert variables to datetime."""
        # convert datetime variables
        datetime_df = pd.concat(
            [
                pd.to_datetime(
                    X[variable],
                    dayfirst=self.dayfirst,
                    yearfirst=self.yearfirst,
                    utc=self.utc,
                    format=self.format,
                )
                for variable in set(self.variables_ + self.reference_)
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
        return datetime_df

    def _sub(self, dt_df: pd.DataFrame):
        """make datetime subtraction"""
        new_df = pd.DataFrame()
        for reference in self.reference_:
            new_varnames = [f"{var}_sub_{reference}" for var in self.variables_]
            new_df[new_varnames] = (
                dt_df[self.variables_]
                .sub(dt_df[reference], axis=0)
                .div(np.timedelta64(1, self.output_unit).astype("timedelta64[ns]"))
            )

        if self.new_variables_names is not None:
            new_df.columns = self.new_variables_names

        return new_df

    def _get_new_features_name(self) -> List:
        """Return names of the created features."""
        if self.new_variables_names is not None:
            feature_names = self.new_variables_names
        else:
            feature_names = [
                f"{var}_sub_{reference}"
                for reference in self.reference_
                for var in self.variables_
            ]
        return feature_names
