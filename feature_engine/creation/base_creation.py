from typing import Optional

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
    _check_param_missing_values,
)
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


class BaseCreation(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """Shared set-up, checks and methods across creation transformers."""

    def __init__(
        self,
        missing_values: str = "raise",
        drop_original: bool = False,
    ) -> None:

        _check_param_missing_values(missing_values)
        _check_param_drop_original(drop_original)

        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe. The variable_handling and dataframe_checks
        # helpers below detect the dataframe backend themselves, so keep passing
        # them the native X rather than the narwhals frame check_X returns.
        check_X(X)

        # check variables are numerical
        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        if hasattr(self, "reference"):
            check_numerical_variables(X, self.reference)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)
            if hasattr(self, "reference"):
                _check_contains_na(X, self.reference)
                _check_contains_inf(X, self.reference)

        # save input features
        if nwd.is_pandas_dataframe(X) is True:
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = nw.from_native(X, eager_only=True).columns

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def _check_transform_input_and_state(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Common input and transformer checks.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe
            The dataframe with the original variables plus the new variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe. As in fit, the checks below detect the
        # backend themselves, so keep working with the native X.
        check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)
            if hasattr(self, "reference"):
                _check_contains_na(X, self.reference)
                _check_contains_inf(X, self.reference)

        # reorder variables to match train set
        if nwd.is_pandas_dataframe(X) is True:
            X = X[self.feature_names_in_]
        else:
            X = nw.from_native(X, eager_only=True).select(
                self.feature_names_in_
            ).to_native()

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "skip"
        # Tests that are OK to fail:
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"][
            "check_fit2d_1feature"
        ] = "this transformer works with datasets that contain at least 2 variables. \
        Otherwise, there is nothing to combine"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
