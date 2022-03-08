# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union, Callable

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine.docstrings import (
    Substitution,
    _drop_original_docstring,
    _feature_names_in_docstring,
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _missing_values_docstring,
    _n_features_in_docstring,
    _variables_numerical_docstring,
)
from feature_engine.timeseries.forecasting.base_forecast import BaseForecast
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
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
class WindowFeatures(BaseForecast):
    """
    WindowFeatures adds "window" features to a dataframe. A window feature is where we
    compute statistics (e.g., mean, min, max, etc.) using a window over the past data.
    For example, the mean value of the previous 3 months of data is a window feature.
    The maximum value of the previous three rows of data is another window feature.

    WindowFeatures uses the functions `rolling()` and `expanding` from pandas to create
    rolling or expanding windows. It also uses pandas' method `agg` to perform multiple
    calculations within those windows. For all supported aggregation functions, see
    `Rolling window functions
    <https://pandas.pydata.org/docs/reference/window.html#api-functions-rolling>`_.

    And finally, it uses the function 'shift()' from pandas, to allocate the value in
    the correct row.

    To be compatible with WindowFeatures, the dataframe's index must have unique values
    and no NaN.

    WindowFeatures works only with numerical variables. You can pass a list of variables
    to use as input for the windows. Alternatively, WindowFeatures will automatically
    select all numerical variables in the training set.

    More details in the :ref:`User Guide <window_features>`.

    Parameters
    ----------
    {variables}

    window: int, offset, or BaseIndexer subclass, default=3.
        Size of the moving window. If an integer, the fixed number of observations used
        for each window. If an offset, the time period of each window. Each window will
        be a variable sized based on the observations included in the time-period. It
        can also take a function. See parameter `windows` in the pandas `rolling()`
        documentation. It is the same functionality.

    min_periods: int, default None.
        Minimum number of observations in window required to have a value; otherwise,
        result is np.nan.

    window_type: string, default="rolling"
        Whether to create rolling or expanding windows. Takes values "rolling" and
        "expanding".

    functions: list of strings, default = ['mean']
        The functions to apply within the window. Valid functions can be found
        `here <https://pandas.pydata.org/docs/reference/window.html>`_.

    periods: int, list of ints, default=1
        Number of periods to shift. Can be a positive integer. See param `periods` in
        pandas `shift`.

    freq: str, list of str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        pandas `shift()`.

    sort_index: bool, default=True
        Whether to order the index of the dataframe before creating the lag features.

    {missing_values}

    {drop_original}

    Attributes
    ----------
    variables_:
        The group of variables that will be lagged.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    transform:
        Add window features.

    {fit_transform}

    See Also
    --------
    pandas.rolling
    pandas.expanding
    pandas.agg
    pandas.shift
    """

    def __init__(
            self,
            variables: Union[None, int, str, List[Union[str, int]]] = None,
            window: Union[str, int, Callable] = 3,
            min_periods: int = None,
            window_type: str = "rolling",
            functions: List[str] = ["mean"],
            periods: int = 1,
            freq: str = None,
            sort_index: bool = True,
            missing_values: str = "raise",
            drop_original: bool = False,
    ) -> None:

        if not isinstance(periods, int) or periods <= 0:
            raise ValueError(
                f"periods must be a positive integer. Got {periods} instead."
            )

        if window_type not in ["rolling", "expanding"]:
            raise ValueError(
                "window_type takes only values 'rolling' or 'expanding'. "
                f"Got {window_type} instead."
            )

        super.__init__(variables, missing_values, drop_original)

        self.window = window
        self.min_periods = min_periods
        self.window_type = window_type
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
        X = super().transform(X)

        tmp = (X[self.variables_]
               .rolling(window=self.window)
               .agg(self.functions)
               .shift(periods=self.periods, freq=self.freq)
               )

        tmp.columns = self.get_feature_names_out(self.variables_)

        X = X.merge(tmp, left_index=True, right_index=True, how="left")

        if self.drop_original:
            X = X.drop(self.variables_, axis=1)

        return X

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features: list, default=None
            Input features. If `input_features` is `None`, then the names of all the
            variables in the transformed dataset (original + new variables) is returned.
            Alternatively, only the names for the lag features derived from
            input_features will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        # create names for all window features or just the indicated ones.
        if input_features is None:
            input_features_ = self.variables_
        else:
            if not isinstance(input_features, list):
                raise ValueError(
                    f"input_features must be a list. Got {input_features} instead."
                )
            if any([f for f in input_features if f not in self.variables_]):
                raise ValueError(
                    "Some features in input_features were not transformed. This method only "
                    "provides the names of the transform features with this method."
                )
            # create just indicated window features
            input_features_ = input_features

        if self.freq is not None:
            feature_names = [
                str(feature) + f"_window_{self.window}_freq_{self.freq}"
                for feature in input_features_
            ]
        else:
            feature_names = [
                str(feature) + f"_window_{self.window}_periods_{self.periods}"
                for feature in input_features_
            ]

        # return names of all variables if input_features is None
        if input_features is None:
            if self.drop_original is True:
                # removes names of variables to drop
                feature_names = [
                    f for f in self.feature_names_in_ if f not in self.variables_
                ]
            else:
                feature_names = self.feature_names_in_ + feature_names

        return feature_names
