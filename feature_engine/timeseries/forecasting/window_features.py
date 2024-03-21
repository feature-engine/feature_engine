from typing import Callable, List, Union

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
class WindowFeatures(BaseForecastTransformer):
    """
    WindowFeatures adds new features to a dataframe based on window operations. Window
    operations are operations that perform an aggregation over a sliding partition of
    past values. A window feature is, in other words, a feature created after computing
    statistics (e.g., mean, min, max, etc.) using a window over the past data. For
    example, the mean value of the previous 3 months of data is a window feature. The
    maximum value of the previous three rows of data is another window feature.

    WindowFeatures uses pandas functions `rolling()`, `agg()` and `shift()`. With
    `rolling()`, it creates rolling windows. With `agg()` it applies multiple functions
    within those windows. With `shift()` it allocates the values to the correct rows.

    For supported aggregation functions, see Rolling Window
    `Functions <https://pandas.pydata.org/docs/reference/window.html>`_.

    With pandas `rolling()` we can perform rolling operations over 1 window size at a
    time. WindowFeatures builds on top of pandas `rolling()` in that new features can
    be derived from multiple window sizes, and the created features will be
    automatically concatenated to the original dataframe.

    To be compatible with WindowFeatures, the dataframe's index must have unique values
    and no missing data.

    WindowFeatures works only with numerical variables. You can pass a list of variables
    to use as input for the windows. Alternatively, WindowFeatures will automatically
    select all numerical variables in the training set.

    More details in the :ref:`User Guide <window_features>`.

    Parameters
    ----------
    {variables}

    window: int, offset, BaseIndexer subclass, or list, default=3
        Size of the moving window. If an integer, the fixed number of observations used
        for each window. If an offset (recommended), the time period of each window. It
        can also take a function. See parameter `windows` in pandas `rolling()`
        documentation for more details.

        In addition to pandas normal input values, `window` can also take a list with
        the above specified values, in which case, features will be created for each
        one of the windows specified in the list.

    min_periods: int, default None.
        Minimum number of observations in the window required to have a value;
        otherwise, the result is np.nan. See parameter `min_periods` in pandas
        `rolling()` documentation for more details.

    functions: string or list of strings, default = 'mean'
        The functions to apply within the window. Valid functions can be found
        `here <https://pandas.pydata.org/docs/reference/window.html>`_.

    periods: int, list of ints, default=1
        Number of periods to shift. Can be a positive integer. See param `periods` in
        pandas `shift()`.

    freq: str, list of str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        pandas `shift()`.

    sort_index: bool, default=True
        Whether to order the index of the dataframe before creating the features.

    {missing_values}

    {drop_original}

    drop_na: bool, default=False.
        Whether the NAN introduced in the lag features should be removed.

    Attributes
    ----------
    variables_:
        The group of variables that will be used to create the window features.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    transform:
        Add window features.

    transform_x_y:
        Remove rows with missing data from X and y.

    {fit_transform}

    See Also
    --------
    pandas.rolling
    pandas.aggregate
    pandas.shift

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import WindowFeatures
    >>> X = pd.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10]
    >>>                     ))
    >>> wf = WindowFeatures(window = 2)
    >>> wf.fit_transform(X)
             date  x1  x2  x1_window_2_mean  x2_window_2_mean
    0  2022-09-18   1   6               NaN               NaN
    1  2022-09-19   2   7               NaN               NaN
    2  2022-09-20   3   8               1.5               6.5
    3  2022-09-21   4   9               2.5               7.5
    4  2022-09-22   5  10               3.5               8.5
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        window: Union[str, int, Callable, List[int], List[str]] = 3,
        min_periods: Union[int, None] = None,
        functions: Union[str, List[str]] = "mean",
        periods: int = 1,
        freq: Union[str, None] = None,
        sort_index: bool = True,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:

        if isinstance(window, list) and len(window) != len(set(window)):
            raise ValueError(f"There are duplicated windows in the list: {window}")

        if not isinstance(functions, (str, list)) or not all(
            isinstance(val, str) for val in functions
        ):
            raise ValueError(
                f"functions must be a string or a list of strings. "
                f"Got {functions} instead."
            )
        if isinstance(functions, list) and len(functions) != len(set(functions)):
            raise ValueError(f"There are duplicated functions in the list: {functions}")

        if not isinstance(periods, int) or periods < 1:
            raise ValueError(
                f"periods must be a positive integer. Got {periods} instead."
            )

        super().__init__(variables, missing_values, drop_original, drop_na)

        self.window = window
        self.min_periods = min_periods
        self.functions = functions
        self.periods = periods
        self.freq = freq
        self.sort_index = sort_index

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds window features.

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

        if isinstance(self.window, list):
            df_ls = []
            for win in self.window:
                tmp = (
                    X[self.variables_]
                    .rolling(window=win)
                    .agg(self.functions)
                    .shift(periods=self.periods, freq=self.freq)
                )
                df_ls.append(tmp)
            tmp = pd.concat(df_ls, axis=1)

        else:
            tmp = (
                X[self.variables_]
                .rolling(window=self.window)
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
        """Get names of the lag features."""

        if not isinstance(self.functions, list):
            functions_ = [self.functions]
        else:
            functions_ = self.functions

        if isinstance(self.window, list):
            feature_names = [
                f"{feature}_window_{win}_{agg}"
                for win in self.window
                for feature in self.variables_
                for agg in functions_
            ]
        else:
            feature_names = [
                f"{feature}_window_{self.window}_{agg}"
                for feature in self.variables_
                for agg in functions_
            ]

        return feature_names
