from typing import List, Optional, Union

from narwhals.typing import IntoDataFrame, IntoSeries
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
from feature_engine.dataframe_helpers import subset_df
from feature_engine.tags import _return_tags
from feature_engine.timeseries.forecasting._forecasting_helpers import (
    check_index,
    sort_by_index,
)
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

    With pandas, the rows are ordered in time by the dataframe's index. Other
    dataframes, like polars, have no index: their rows are taken in the order given.

    Subclasses define the parameters `freq` and `sort_index`.

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
        X = check_X(X)

        # We need the dataframes to have unique values in the index and no missing data.
        # Otherwise, when we merge the new features we will duplicate rows.
        check_index(X, self.freq)

        if self.variables is None:
            self.variables_ = find_numerical_variables(
                X, return_empty=self.return_empty
            )
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def _check_transform_input_and_state(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Common checks performed before the feature transformation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X: pandas or narwhals dataframe of shape = [n_samples, n_features]
            The data to transform, with the variables in the same order as in the
            train set. With pandas, the rows are sorted by the index if
            `sort_index=True`.
        """
        check_is_fitted(self)
        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)
        check_index(X, self.freq)

        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        X = subset_df(X, self.feature_names_in_)
        if self.sort_index is True:
            X = sort_by_index(X)

        return X

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
