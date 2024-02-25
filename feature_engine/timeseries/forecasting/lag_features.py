# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring, _n_features_in_docstring)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _drop_original_docstring, _missing_values_docstring,
    _variables_numerical_docstring)
from feature_engine._docstrings.methods import (_fit_not_learn_docstring,
                                                _fit_transform_docstring)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.timeseries.forecasting.base_forecast_transformers import \
    BaseForecastTransformer


@Substitution(
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class LagFeatures(BaseForecastTransformer):
    """
    LagFeatures adds lag features to the dataframe. A lag feature is a feature with
    information about a prior time step.

    LagFeatures has the same functionality as pandas `shift()` with the exception that
    only one of `periods` or `freq` can be indicated at a time. LagFeatures builds on
    top of pandas `shift()` in that multiple lags can be created at the same time and
    the features with names will be concatenated to the original dataframe.

    To be compatible with LagFeatures, the dataframe's index must have unique values
    and no NaN.

    LagFeatures works only with numerical variables. You can pass a list of variables
    to lag. Alternatively, LagFeatures will automatically select and lag all numerical
    variables found in the training set.

    More details in the :ref:`User Guide <lag_features>`.

    Parameters
    ----------
    {variables}

    periods: int, list of ints, default=1
        Number of periods to shift. Can be a positive integer or list of positive
        integers. If list, features will be created for each one of the periods in the
        list. If the parameter `freq` is specified, `periods` will be ignored.

    freq: str, list of str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        pandas `shift()`. It is the same functionality. If freq is a list, lag features
        will be created for each one of the frequency values in the list. If freq is not
        None, then this parameter overrides the parameter `periods`.

    sort_index: bool, default=True
        Whether to order the index of the dataframe before creating the lag features.

    {missing_values}

    {drop_original}

    group_by_variables: str, list of str, default=None
            variable of list of variables to create lag features based on.

    Attributes
    ----------
    variables_:
        The group of variables that will be lagged.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    transform:
        Add lag features.

    See Also
    --------
    pandas.shift

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import LagFeatures
    >>> X = pd.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10]
    >>>                     ))
    >>> lf = LagFeatures(periods=[1,2])
    >>> lf.fit_transform(X)
                date  x1  x2  x1_lag_1  x2_lag_1  x1_lag_2  x2_lag_2
    0  2022-09-18   1   6       NaN       NaN       NaN       NaN
    1  2022-09-19   2   7       1.0       6.0       NaN       NaN
    2  2022-09-20   3   8       2.0       7.0       1.0       6.0
    3  2022-09-21   4   9       3.0       8.0       2.0       7.0
    4  2022-09-22   5  10       4.0       9.0       3.0       8.0

    create lags based on other variables.
    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import LagFeatures
    >>> X = pd.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10],
    >>>                       x3 = ['a','b','a','b','a']
    >>>                     ))
    >>> lf = LagFeatures(periods=[1,2], group_by_variables='x3')
    >>> lf.fit_transform(X)
              date  x1  x2 x3  x1_lag_1  x2_lag_1  x1_lag_2  x2_lag_2
    0  2022-09-18   1   6  a       NaN       NaN       NaN       NaN
    1  2022-09-19   2   7  b       NaN       NaN       NaN       NaN
    2  2022-09-20   3   8  a       1.0       6.0       NaN       NaN
    3  2022-09-21   4   9  b       2.0       7.0       NaN       NaN
    4  2022-09-22   5  10  a       3.0       8.0       1.0       6.0
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        periods: Union[int, List[int]] = 1,
        freq: Union[str, List[str], None] = None,
        sort_index: bool = True,
        missing_values: str = "raise",
        drop_original: bool = False,
        group_by_variables: Optional[Union[str, List[str]]] = None,
    ) -> None:

        if not (
            isinstance(periods, int)
            and periods > 0
            or isinstance(periods, list)
            and all(isinstance(num, int) and num > 0 for num in periods)
        ):

            raise ValueError(
                "periods must be an integer or a list of positive integers. "
                f"Got {periods} instead."
            )
        if isinstance(periods, list) and len(periods) != len(set(periods)):
            raise ValueError(f"There are duplicated periods in the list: {periods}")

        if isinstance(freq, list) and len(freq) != len(set(freq)):
            raise ValueError(f"There are duplicated freq values in the list: {freq}")

        if not isinstance(sort_index, bool):
            raise ValueError(
                "sort_index takes values True and False." f"Got {sort_index} instead."
            )

        super().__init__(variables, missing_values, drop_original, group_by_variables)

        self.periods = periods
        self.freq = freq
        self.sort_index = sort_index

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds lag features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features + lag_features]
            The dataframe with the original plus the new variables.
        """
        # Common dataframe checks and setting up.
        X = self._check_transform_input_and_state(X)

        # if freq is not None, it overrides periods.
        if self.freq is not None:

            if isinstance(self.freq, list):
                df_ls = []
                for fr in self.freq:
                    if self.group_by_variables:
                        tmp = X.groupby(self.group_by_variables)[self.variables_].shift(
                            freq=fr,
                        )
                    else:
                        tmp = X[self.variables_].shift(
                            freq=fr,
                            axis=0,
                        )
                    df_ls.append(tmp)
                tmp = pd.concat(df_ls, axis=1)

            else:
                if self.group_by_variables:
                    tmp = X.groupby(self.group_by_variables)[self.variables_].shift(
                        freq=self.freq,
                    )
                else:
                    tmp = X[self.variables_].shift(
                        freq=self.freq,
                        axis=0,
                    )

        else:
            if isinstance(self.periods, list):
                df_ls = []
                for pr in self.periods:
                    if self.group_by_variables:
                        tmp = X.groupby(self.group_by_variables)[self.variables_].shift(
                            periods=pr,
                        )
                    else:
                        tmp = X[self.variables_].shift(
                            periods=pr,
                            axis=0,
                        )
                    df_ls.append(tmp)
                tmp = pd.concat(df_ls, axis=1)

            else:
                if self.group_by_variables:
                    tmp = X.groupby(self.group_by_variables)[self.variables_].shift(
                        periods=self.periods,
                    )
                else:
                    tmp = X[self.variables_].shift(
                        periods=self.periods,
                        axis=0,
                    )

        tmp.columns = self._get_new_features_name()

        X = X.merge(tmp, left_index=True, right_index=True, how="left")

        if self.drop_original:
            X = X.drop(self.variables_, axis=1)

        return X

    def _get_new_features_name(self) -> List:
        """Get names of the lag features."""

        # create the names for the lag features
        if isinstance(self.freq, list):
            feature_names = [
                f"{feature}_lag_{fr}" for fr in self.freq for feature in self.variables_
            ]
        elif self.freq is not None:
            feature_names = [
                f"{feature}_lag_{self.freq}" for feature in self.variables_
            ]
        elif isinstance(self.periods, list):
            feature_names = [
                f"{feature}_lag_{pr}"
                for pr in self.periods
                for feature in self.variables_
            ]
        else:
            feature_names = [
                f"{feature}_lag_{self.periods}" for feature in self.variables_
            ]

        return feature_names
