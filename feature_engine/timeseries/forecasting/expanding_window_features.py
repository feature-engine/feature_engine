# Author: Kishan Manani
# License: BSD 3 clause

from __future__ import annotations

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import (
    _variables_numerical_docstring,
    _drop_original_docstring,
    _missing_values_docstring,
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

    {fit_transform}

    See Also
    --------
    pandas.expanding
    pandas.aggregate
    pandas.shift
    """

    def __init__(
        self,
        variables: None | int | str | list[str | int] = None,
        min_periods: int | None = None,
        functions: str | list[str] = "mean",
        periods: int = 1,
        freq: str = None,
        sort_index: bool = True,
        missing_values: str = "raise",
        drop_original: bool = False,
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

        super().__init__(variables, missing_values, drop_original)

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
        X = super().transform(X)

        tmp = (
            X[self.variables_]
            .expanding(min_periods=self.min_periods)
            .agg(self.functions)
            .shift(periods=self.periods, freq=self.freq)
        )

        tmp.columns = self.get_feature_names_out(self.variables_)

        X = X.merge(tmp, left_index=True, right_index=True, how="left")

        if self.drop_original:
            X = X.drop(self.variables_, axis=1)

        return X

    def get_feature_names_out(self, input_features: list | None = None) -> list:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features: list, default=None
            Input features. If `input_features` is `None`, then the names of all the
            variables in the transformed dataset (original + new variables) is returned.
            Alternatively, only the names for the window features derived from
            input_features will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        # create names for all expanding window features or just the indicated ones.
        if input_features is None:
            input_features_ = self.variables_
        else:
            if not isinstance(input_features, list):
                raise ValueError(
                    f"input_features must be a list. Got {input_features} instead."
                )
            if any([f for f in input_features if f not in self.variables_]):
                raise ValueError(
                    "Some of the indicated features were not used to create window "
                    "features."
                )
            # create just indicated window features
            input_features_ = input_features

        if not isinstance(self.functions, list):
            functions_ = [self.functions]
        else:
            functions_ = self.functions

        feature_names = [
            f"{feature}_expanding_{agg}"
            for feature in input_features_
            for agg in functions_
        ]

        # return names of all variables if input_features is None
        if input_features is None:
            if self.drop_original is True:
                # removes names of variables to drop
                original = [
                    f for f in self.feature_names_in_ if f not in self.variables_
                ]
                feature_names = original + feature_names
            else:
                feature_names = self.feature_names_in_ + feature_names

        return feature_names
