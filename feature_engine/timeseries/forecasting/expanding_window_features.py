# Author: Kishan Manani
# License: BSD 3 clause

from __future__ import annotations

from typing import List

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _drop_original_docstring,
    _missing_values_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.timeseries.forecasting.base_forecast_transformers import (
    BaseForecastTransformer,
)


@Substitution(
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class ExpandingWindowFeatures(BaseForecastTransformer):
    """
    ExpandingWindowFeatures adds new features to a dataframe based on expanding window
    operations. Expanding window operations are operations that perform an
    aggregation over an expanding window of all past values relative to the
    value of interest. An expanding window feature is, in other words, a feature
    created after computing statistics (e.g., mean, min, max, etc.) using a
    window over all the past data. For example, the mean value of all months
    prior to the month of interest is an expanding window feature.

    ExpandingWindowFeatures uses the pandas' functions `expanding()`, `agg()` and
    `shift()`. With `expanding()`, it creates expanding windows. With `agg()` it
    applies multiple functions within those windows. With 'shift()' it allocates
    the values to the correct rows.

    For supported aggregation functions, see Expanding Window
    `Functions
    <https://pandas.pydata.org/docs/reference/window.html#expanding-window-functions>`_.

    To be compatible with ExpandingWindowFeatures, the dataframe's index must
    have unique values and no NaN.

    ExpandingWindowFeatures works only with numerical variables. You can pass a
    list of variables to use as input for the expanding window. Alternatively,
    ExpandingWindowFeatures will automatically select all numerical variables
    in the training set.

    More details in the :ref:`User Guide <expanding_window_features>`.

    Parameters
    ----------
    {variables}

    min_periods: int, default None.
        Minimum number of observations in window required to have a value;
        otherwise, result is np.nan. See parameter `min_periods` in the pandas
        `expanding()` documentation for more details.

    functions: str, list of str, default = 'mean'
        The functions to apply within the window. Valid functions can be found
        `here <https://pandas.pydata.org/docs/reference/window.html>`_.

    periods: int, list of ints, default=1
        Number of periods to shift. Can be a positive integer. See param `periods` in
        pandas `shift`.

    freq: str, list of str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        pandas `shift()`.

    sort_index: bool, default=True
        Whether to order the index of the dataframe before creating the
        expanding window feature.

    {missing_values}

    {drop_original}

    drop_na: bool, default=False.
        Whether the NAN introduced in the created features should be removed.


    Attributes
    ----------
    variables_:
        The group of variables that will be used to create the expanding window
        features.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    transform:
        Add expanding window features.

    transform_x_y:
        Remove rows with missing data from X and y.

    {fit_transform}

    See Also
    --------
    pandas.expanding
    pandas.aggregate
    pandas.shift

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import ExpandingWindowFeatures
    >>> X = pd.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10]
    >>>                     ))
    >>> ewf = ExpandingWindowFeatures()
    >>> ewf.fit_transform(X)
             date  x1  x2  x1_expanding_mean  x2_expanding_mean
    0  2022-09-18   1   6                NaN                NaN
    1  2022-09-19   2   7                1.0                6.0
    2  2022-09-20   3   8                1.5                6.5
    3  2022-09-21   4   9                2.0                7.0
    4  2022-09-22   5  10                2.5                7.5
    """

    def __init__(
        self,
        variables: None | int | str | list[str | int] = None,
        min_periods: int | None = None,
        functions: str | list[str] = "mean",
        periods: int = 1,
        freq: str | None = None,
        sort_index: bool = True,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:

        if not isinstance(functions, (str, list)) or not all(
            isinstance(val, str) for val in functions
        ):
            raise ValueError(
                f"functions must be a list of strings or a string."
                f"Got {functions} instead."
            )
        if isinstance(functions, list) and len(functions) != len(set(functions)):
            raise ValueError(f"There are duplicated functions in the list: {functions}")

        if not isinstance(periods, int) or periods < 0:
            raise ValueError(
                f"periods must be a non-negative integer. Got {periods} instead."
            )

        super().__init__(variables, missing_values, drop_original, drop_na)

        self.min_periods = min_periods
        self.functions = functions
        self.periods = periods
        self.freq = freq
        self.sort_index = sort_index

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds expanding window features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features + window_features]
            The dataframe with the original plus the new variables.
        """
        # Common dataframe checks and setting up.
        X = self._check_transform_input_and_state(X)

        tmp = (
            X[self.variables_]
            .expanding(min_periods=self.min_periods)
            .agg(self.functions)
            .shift(periods=self.periods, freq=self.freq)
        )

        tmp.columns = self._get_new_features_name()

        X = X.merge(tmp, left_index=True, right_index=True, how="left")

        if self.drop_original:
            X = X.drop(self.variables_, axis=1)

        if self.drop_na:
            X = X.dropna(subset=tmp.columns, axis=0)

        return X

    def _get_new_features_name(self) -> List:
        """Get names of the window features."""

        if not isinstance(self.functions, list):
            functions_ = [self.functions]
        else:
            functions_ = self.functions

        feature_names = [
            f"{feature}_expanding_{agg}"
            for feature in self.variables_
            for agg in functions_
        ]

        return feature_names
