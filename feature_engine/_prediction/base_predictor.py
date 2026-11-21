from typing import List, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
    check_X_y,
)
from feature_engine.dataframe_helpers import (
    map_categories_in_df,
    subset_df,
    to_numpy,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)
from feature_engine.encoding._helper_functions import TARGET_NAME, add_target_to_X
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import find_categorical_and_numerical_variables


class BaseTargetMeanEstimator(BaseEstimator):
    """
    Calculates the mean target value per category or per bin of a variable or group of
    variables. Works with numerical and categorical variables. If variables are
    numerical, the values are first sorted into bins of equal-width or equal-frequency.

    Parameters
    ----------
    variables: list, default=None
        The list of input variables. If None, the estimator will use all variables as
        input features (except datetime).

    bins: int, default=5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default='equal_width'
        Whether the bins should be of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    Attributes
    ----------
    variables_categorical_:
        The group of categorical input variables that will be used for prediction.

    variables_numerical_:
        The group of numerical input variables that will be used for prediction.

    binner_dict_:
         Dictionary with the interval limits per numerical variable.

    encoder_dict_:
        Dictionary with the mean target value per category or interval, per variable.

    n_features_in_:
        The number of features in the train set used in fit.

    feature_names_in_:
        List with the names of features seen during `fit`.

    See Also
    --------
    feature_engine.encoding.MeanEncoder
    feature_engine.discretisation.EqualWidthDiscretiser
    feature_engine.discretisation.EqualFrequencyDiscretiser

    References
    ----------
    Adapted from:

    .. [1] Miller, et al. "Predicting customer behaviour: The University of Melbourne’s
        KDD Cup report". JMLR Workshop and Conference Proceeding. KDD 2009
        http://proceedings.mlr.press/v7/miller09/miller09.pdf
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        bins: int = 5,
        strategy: str = "equal_width",
    ):

        if not isinstance(bins, int) or bins < 1:
            raise ValueError(f"bins must be a positive integer. Got {bins} instead.")

        if not isinstance(strategy, str) or strategy not in [
            "equal_width",
            "equal_frequency",
        ]:
            raise ValueError(
                "strategy takes only values 'equal_width' or 'equal_frequency'. "
                f"Got {strategy} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.bins = bins
        self.strategy = strategy

    def fit(
        self,
        X: IntoDataFrame,
        y: Union[IntoSeries, np.ndarray, List],
    ):
        """
        Learn the mean target value per category or bin.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: Series, numpy array or list of shape = [n_samples,]
            The target variable.
        """
        X, y = check_X_y(X, y)

        (
            variables_categorical_,
            variables_numerical_,
        ) = find_categorical_and_numerical_variables(X, self.variables)

        _check_contains_na(X, variables_numerical_ + variables_categorical_)
        _check_contains_inf(X, variables_numerical_)

        # adding a column to a wide pandas dataframe is slow, so take the variables.
        nw_Xy = add_target_to_X(
            nw.from_native(subset_df(X, variables_categorical_), eager_only=True), y
        )
        target = nw_Xy.get_column(TARGET_NAME)

        encoder_dict_ = {}
        binner_dict_ = {}
        bin_means = {}

        if len(variables_numerical_) > 0:
            # the discretiser checks X again, and expects the user's dataframe.
            discretiser = self._make_discretiser(variables_numerical_)
            discretiser.fit(nw.to_native(X, pass_through=True))
            binner_dict_ = discretiser.binner_dict_
            target_values = target.to_numpy().astype(float)
            arrays = to_numpy(X, variables_numerical_)
            for var, values in zip(variables_numerical_, arrays):
                edges = np.asarray(binner_dict_[var], dtype=float)
                codes, _ = discretiser._digitize(values, edges)
                n_bins = len(edges) - 1
                counts = np.bincount(codes, minlength=n_bins)
                sums = np.bincount(codes, weights=target_values, minlength=n_bins)
                # NaN marks the bins without training observations, which _predict
                # treats as unseen values.
                with np.errstate(invalid="ignore"):
                    bin_means[var] = sums / counts
                labels = discretiser._format_bin_labels(edges, discretiser.precision)
                encoder_dict_[var] = {
                    labels[code]: bin_means[var][code].item()
                    for code in np.flatnonzero(counts)
                }
            self._discretiser = discretiser

        for var in variables_categorical_:
            # group_by() needs string names; pandas columns can be integers.
            category = nw_Xy.get_column(var).alias("__category__")
            stats = (
                category.to_frame()
                .with_columns(target)
                .group_by("__category__")
                .agg(nw.col(TARGET_NAME).mean())
            )
            encoder_dict_[var] = dict(
                zip(stats["__category__"].to_list(), stats[TARGET_NAME].to_list())
            )

        self.variables_categorical_ = variables_categorical_
        self.variables_numerical_ = variables_numerical_
        self.binner_dict_ = binner_dict_
        self.encoder_dict_ = encoder_dict_
        self._bin_means = bin_means
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def _make_discretiser(self, variables: List[Union[str, int]]):
        """
        Instantiate the EqualWidthDiscretiser or EqualFrequencyDiscretiser.
        """
        if self.strategy == "equal_width":
            discretiser = EqualWidthDiscretiser(bins=self.bins, variables=variables)
        else:
            discretiser = EqualFrequencyDiscretiser(q=self.bins, variables=variables)

        return discretiser

    def _predict(self, X: IntoDataFrame) -> np.ndarray:
        """
        Predict using the average of the target mean value across variables.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y_pred: numpy array of shape = (n_samples, )
            The mean target value per observation.
        """
        check_is_fitted(self)
        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)
        _check_contains_na(X, self.variables_numerical_ + self.variables_categorical_)
        _check_contains_inf(X, self.variables_numerical_)

        predictions = np.zeros(X.shape[0])

        unseen = []
        arrays = to_numpy(X, self.variables_numerical_)
        for var, values in zip(self.variables_numerical_, arrays):
            codes, _ = self._discretiser._digitize(
                values, np.asarray(self.binner_dict_[var], dtype=float)
            )
            encoded = self._bin_means[var][codes]
            if np.isnan(encoded).any():
                unseen.append(var)
            predictions += encoded
        self._raise_if_unseen(unseen)

        if len(self.variables_categorical_) > 0:
            X_encoded = map_categories_in_df(
                subset_df(X, self.variables_categorical_),
                {var: self.encoder_dict_[var] for var in self.variables_categorical_},
            )
            arrays = to_numpy(X_encoded, self.variables_categorical_)
            for var, encoded in zip(self.variables_categorical_, arrays):
                if np.isnan(encoded).any():
                    unseen.append(var)
                predictions += encoded
            self._raise_if_unseen(unseen)

        return predictions / (
            len(self.variables_numerical_) + len(self.variables_categorical_)
        )

    def _raise_if_unseen(self, variables: List[Union[str, int]]):
        if len(variables) > 0:
            raise ValueError(
                "During the encoding, NaN values were introduced in the feature(s) "
                f"{', '.join(str(var) for var in variables)}."
            )

    def _more_tags(self):
        return _return_tags()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
