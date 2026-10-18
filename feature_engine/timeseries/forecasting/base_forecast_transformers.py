from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import (
    GetFeatureNamesOutMixin,
    TransformXyMixin,
)
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _drop_original_docstring,
    _missing_values_docstring,
)
from feature_engine._docstrings.methods import _fit_not_learn_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


@Substitution(
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    fit=_fit_not_learn_docstring,
    n_features_in_=_n_features_in_docstring,
)
class BaseForecastTransformer(
    TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin, TransformXyMixin
):
    """
    Shared methods across time-series forecasting transformers.

    Parameters
    ----------
    variables: str, int, or list of strings or integers, default=None.
        The variables to use to create the new features.

    {missing_values}

    {drop_original}

    drop_na: bool, default=False.
        Whether the NAN introduced in the created features should be removed.

    Attributes
    ----------
    {feature_names_in_}

    {n_features_in_}

    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
        groupby=None,
        sort_by=None,
    ) -> None:

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if not isinstance(drop_original, bool):
            raise ValueError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )

        if not isinstance(drop_na, bool):
            raise ValueError(
                "drop_na takes only boolean values True and False. "
                f"Got {drop_na} instead."
            )

        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.missing_values = missing_values
        self.drop_original = drop_original
        self.drop_na = drop_na
        self.groupby = _check_variables_input_value(groupby)
        if sort_by is not None and not isinstance(sort_by, (str, int)):
            raise ValueError("sort_by must be a column name or None.")
        if sort_by is not None and groupby is None:
            raise ValueError("sort_by requires groupby.")
        self.sort_by = sort_by

    def _check_index(self, X: pd.DataFrame):
        """
        Check that the index does not have missing data and its values are unique.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataset.
        """
        if X.index.isnull().any():
            raise NotImplementedError(
                "The dataframe's index contains NaN values or missing data. "
                "Only dataframes with complete indexes are compatible with "
                "this transformer."
            )

        if X.index.is_unique is False:
            raise NotImplementedError(
                "The dataframe's index does not contain unique values. "
                "Only dataframes with unique values in the index are "
                "compatible with this transformer."
            )

        return self

    def _check_na_and_inf(self, X: pd.DataFrame):
        """
        Checks that the dataframe does not contain NaN or Infinite values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataset for training or transformation.
        """
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        return self

    def _get_feature_names_in(self, X: pd.DataFrame):
        """
        Finds the number and name of the features in the training set.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataset for training or transformation.
        """

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        return self

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this transformer. You can pass None or y.
        """
        # check input dataframe
        X = check_X(X)

        # We need the dataframes to have unique values in the index and no missing data.
        # Otherwise, when we merge the new features we will duplicate rows.
        self._check_index(X)

        self._check_grouping(X)

        # find or check for numerical variables
        if self.variables is None:
            self.variables_ = find_numerical_variables(
                X, return_empty=self.return_empty
            )
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        if self.groupby is not None:
            excluded = self._group_columns() + (
                [] if self.sort_by is None else [self.sort_by]
            )
            if self.variables is not None and set(self.variables_) & set(excluded):
                raise ValueError("Grouping and sorting columns cannot be variables.")
            self.variables_ = [v for v in self.variables_ if v not in excluded]
            if not self.variables_ and not self.return_empty:
                raise ValueError(
                    "No numerical variables remain after excluding group columns."
                )

        # check if dataset contains na
        if self.missing_values == "raise":
            self._check_na_and_inf(X)

        self._get_feature_names_in(X)

        return self

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Common checks performed before the feature transformation.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.
        """
        # check method fit has been called
        check_is_fitted(self)

        # check if 'X' is a dataframe
        X = check_X(X)

        # check if input data contains the same number of columns as the fitted
        # dataframe.
        _check_X_matches_training_df(X, self.n_features_in_)

        # Dataframes must have unique values in the index and no missing data.
        # Otherwise, when we merge the created features we will duplicate rows.
        self._check_index(X)
        self._check_grouping(X)

        # check if dataset contains na
        if self.missing_values == "raise":
            self._check_na_and_inf(X)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        if self.sort_by is not None:
            X = X.sort_values(self.sort_by, kind="stable")
        elif self.sort_index is True:
            X.sort_index(inplace=True)

        return X

    def _group_columns(self):
        return self.groupby if isinstance(self.groupby, list) else [self.groupby]

    def _check_grouping(self, X):
        if self.groupby is None:
            return
        columns = self._group_columns()
        required = columns + ([] if self.sort_by is None else [self.sort_by])
        if any(c not in X.columns for c in required):
            raise ValueError("Grouping and sorting columns must be present in X.")
        if X[required].isna().any().any():
            raise ValueError(
                "Grouping and sorting columns cannot contain missing values."
            )
        if self.sort_by is not None and X.duplicated(required).any():
            raise ValueError("sort_by must be unique within each group.")

    def _transform_grouped(self, X, kind):
        """Compute features within groups, then align on group and time together."""
        if not self.variables_:
            return X
        data = X[self.variables_].copy()
        if self.sort_by is not None:
            data.index = pd.Index(X[self.sort_by])
        keys = [X[c].to_numpy() for c in self._group_columns()]
        target_index = pd.MultiIndex.from_arrays(keys + [data.index])
        grouped = data.groupby(keys, sort=False, observed=True)
        levels = list(range(len(keys)))
        parts = []
        if kind == "lag":
            shifts = self.freq if self.freq is not None else self.periods
            shifts = shifts if isinstance(shifts, list) else [shifts]
            for shift in shifts:
                if self.freq is None:
                    part = grouped.shift(shift, fill_value=self.fill_value)
                    part.index = X.index
                else:
                    part = data.copy()
                    part.index = pd.MultiIndex.from_arrays(
                        keys + [data.index.shift(1, freq=shift)]
                    )
                    part = part.reindex(target_index)
                    part.index = X.index
                    if self.fill_value is not None:
                        part = part.fillna(self.fill_value)
                parts.append(part)
        else:
            windows = self.window if kind == "window" else [None]
            windows = windows if isinstance(windows, list) else [windows]
            for window in windows:
                if kind == "window":
                    part = grouped.rolling(
                        window=window, min_periods=self.min_periods
                    ).agg(self.functions)
                else:
                    part = grouped.expanding(min_periods=self.min_periods).agg(
                        self.functions
                    )
                if self.freq is None:
                    part = part.groupby(level=levels, sort=False).shift(self.periods)
                else:
                    arrays = [part.index.get_level_values(i) for i in levels]
                    time = part.index.get_level_values(-1)
                    part.index = pd.MultiIndex.from_arrays(
                        arrays + [time.shift(self.periods, freq=self.freq)]
                    )
                part = part.reindex(target_index)
                part.index = X.index
                parts.append(part)
        features = pd.concat(parts, axis=1)
        features.columns = self._get_new_features_name()
        result = pd.concat([X, features], axis=1)
        if self.drop_original:
            result = result.drop(columns=self.variables_)
        if self.drop_na:
            result = result.dropna(subset=features.columns)
        return result

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "numerical"
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_methods_subset_invariance"
        ] = "LagFeatures is not invariant when applied to a subset. Not sure why yet"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
