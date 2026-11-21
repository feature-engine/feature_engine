from typing import List, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
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
from feature_engine.dataframe_helpers import subset_df, to_numpy
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)
from feature_engine.encoding import MeanEncoder
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

        pipeline = self._make_pipeline(variables_numerical_, variables_categorical_)
        # the pipeline checks X again, and expects the user's dataframe.
        pipeline.fit(nw.to_native(X, pass_through=True), y)

        steps = pipeline.named_steps
        binner_dict_ = {}
        encoder_dict_ = {}
        if len(variables_numerical_) > 0:
            binner_dict_ = dict(steps["discretiser"].binner_dict_)
            encoder_dict_.update(steps["encoder_num"].encoder_dict_)
        if len(variables_categorical_) > 0:
            encoder_dict_.update(steps["encoder_cat"].encoder_dict_)

        self.variables_categorical_ = variables_categorical_
        self.variables_numerical_ = variables_numerical_
        self.binner_dict_ = binner_dict_
        self.encoder_dict_ = encoder_dict_
        self._pipeline = pipeline
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def _make_pipeline(
        self,
        variables_numerical: List[Union[str, int]],
        variables_categorical: List[Union[str, int]],
    ) -> Pipeline:
        """
        Sort the numerical variables into bins, and replace bins and categories by
        the mean target value.
        """
        steps = []
        if len(variables_numerical) > 0:
            steps.append(("discretiser", self._make_discretiser(variables_numerical)))
            encoder = MeanEncoder(variables=variables_numerical, unseen="raise")
            steps.append(("encoder_num", encoder))
        if len(variables_categorical) > 0:
            encoder = MeanEncoder(variables=variables_categorical, unseen="raise")
            steps.append(("encoder_cat", encoder))
        return Pipeline(steps)

    def _make_discretiser(self, variables: List[Union[str, int]]):
        """
        Instantiate the EqualWidthDiscretiser or EqualFrequencyDiscretiser.
        """
        if self.strategy == "equal_width":
            discretiser = EqualWidthDiscretiser(
                bins=self.bins, variables=variables, return_boundaries=True
            )
        else:
            discretiser = EqualFrequencyDiscretiser(
                q=self.bins, variables=variables, return_boundaries=True
            )

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
        X = subset_df(X, self.feature_names_in_)

        X_tr = check_X(self._pipeline.transform(nw.to_native(X, pass_through=True)))
        variables = self.variables_numerical_ + self.variables_categorical_
        # summing in order gives the same values as pandas' mean(axis=1).
        predictions = np.zeros(X_tr.shape[0])
        for values in to_numpy(X_tr, variables):
            predictions += values
        return predictions / len(variables)

    def _more_tags(self):
        return _return_tags()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
