from typing import List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
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
    check_X_y,
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
class BaseForecastTransformer(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    Shared methods across time-series forecasting transformers.

    With pandas, the rows are ordered in time by the dataframe's index. Other
    dataframes, like polars, have no index: their rows are taken in the order given.

    Subclasses define the parameters `freq` and `sort_index`, and the methods
    `_add_features()` and `_get_new_features_name()`.

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
    ) -> None:

        if not isinstance(missing_values, str) or missing_values not in [
            "raise",
            "ignore",
        ]:
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

    def _check_index(self, X: IntoDataFrame):
        """
        Checks that the rows of the dataframe can be ordered in time. With pandas,
        the index must be unique and not contain missing data. Other dataframes have
        no index, so they can't use `freq`.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The dataset.
        """
        if nwd.is_pandas_dataframe(X) is True:
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

        elif self.freq is not None:
            raise NotImplementedError(
                "freq is only supported with pandas dataframes, because it uses the "
                "dataframe's DatetimeIndex. With other dataframes, leave freq=None "
                f"to shift the rows by periods. Got {self.freq} instead."
            )

        return self

    def _check_na_and_inf(self, X: IntoDataFrame):
        """
        Checks that the dataframe does not contain NaN or Infinite values.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The dataset for training or transformation.
        """
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        return self

    def _get_feature_names_in(self, X: IntoDataFrame):
        """
        Finds the number and name of the features in the training set.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The dataset for training or transformation.
        """
        if nwd.is_pandas_dataframe(X) is True:
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = nw.from_native(X, eager_only=True).columns
        self.n_features_in_ = X.shape[1]

        return self

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: Series, default=None
            y is not needed in this transformer. You can pass None or y.
        """
        check_X(X)

        # With pandas, the new features are aligned to the rows through the index,
        # so an index with duplicates or missing data would duplicate rows.
        self._check_index(X)

        if self.variables is None:
            self.variables_ = find_numerical_variables(
                X, return_empty=self.return_empty
            )
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            self._check_na_and_inf(X)

        self._get_feature_names_in(X)

        return self

    def _check_transform_input_and_state(self, X: IntoDataFrame) -> nw.DataFrame:
        """
        Common checks performed before the feature transformation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        nw_X: narwhals dataframe of shape = [n_samples, n_features]
            The data to transform, with the variables in the same order as in the
            train set. With pandas, the rows are sorted by the index if
            `sort_index=True`.
        """
        check_is_fitted(self)

        nw_X = check_X(X)

        _check_X_matches_training_df(nw_X, self.n_features_in_)

        self._check_index(X)

        if self.missing_values == "raise":
            self._check_na_and_inf(X)

        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals, and only pandas has an index to sort.
            X = X[self.feature_names_in_]
            if self.sort_index is True:
                X.sort_index(inplace=True)
            return nw.from_native(X, eager_only=True)

        # without an index, the rows are used in the order given: sort_index is ignored.
        return nw_X.select(self.feature_names_in_)

    def _add_features(self, nw_X: nw.DataFrame) -> nw.DataFrame:
        """
        Adds the new features, named as `_get_new_features_name()` returns, after
        the columns of `nw_X`. Rows must keep their order.

        Parameters
        ----------
        nw_X: narwhals dataframe of shape = [n_samples, n_features]
            The data returned by `_check_transform_input_and_state()`.
        """
        raise NotImplementedError

    def _transform(self, X: IntoDataFrame) -> nw.DataFrame:
        """
        Checks the input, adds the new features and drops the original variables if
        requested. Rows with missing data are not dropped.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.
        """
        nw_X = self._add_features(self._check_transform_input_and_state(X))

        if self.drop_original is True:
            nw_X = nw_X.drop(self.variables_)

        return nw_X

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Adds the new features.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features + new_features]
            The dataframe with the original plus the new variables. If
            `drop_na=True`, rows with missing data in the new features are removed.
        """
        nw_X = self._transform(X)

        if self.drop_na is True:
            new_features = self._get_new_features_name()
            X = nw_X.to_native()
            if nwd.is_pandas_dataframe(X) is True:
                # a numpy mask is faster than pandas' dropna() and narwhals.
                return X[~X[new_features].isna().to_numpy().any(axis=1)]
            nw_X = nw_X.drop_nulls(subset=new_features)

        return nw_X.to_native()

    def transform_x_y(self, X: IntoDataFrame, y: IntoSeries):
        """
        Adds the new features to X and, if `drop_na=True`, removes the rows with
        missing data in the new features from both X and y.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The dataframe to transform.

        y: Series or Dataframe of length = n_samples
            The target variable to transform. Can be multi-output.

        Returns
        -------
        X_new: dataframe
            The transformed dataframe. It may contain less rows than the original
            dataset.

        y_new: Series or DataFrame
            The target variable, with as many rows as those left in X_new.
        """
        _, y = check_X_y(X, y)

        if nwd.is_pandas_dataframe(X) is True:
            # the rows may be sorted by the index, so y follows the index of X.
            X = self.transform(X)
            return X, y.loc[X.index]

        nw_X = self._transform(X)
        new_features = self._get_new_features_name()

        if self.drop_na is False or len(new_features) == 0:
            return nw_X.to_native(), y

        # without an index, a mask of the rows to keep subsets both X and y.
        keep = nw_X.select(
            (
                ~nw.any_horizontal(nw.col(new_features).is_null(), ignore_nulls=True)
            ).alias("__keep__")
        ).get_column("__keep__")
        if nwd.is_into_series(y):
            y = nw.from_native(y, series_only=True).filter(keep).to_native()
        else:
            y = nw.from_native(y, eager_only=True).filter(keep).to_native()

        return nw_X.filter(keep).to_native(), y

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
