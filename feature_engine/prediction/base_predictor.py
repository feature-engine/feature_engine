# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, check_X_y

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)
from feature_engine.encoding import MeanEncoder
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_categorical_and_numerical_variables,
)


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
        Whether the bins should of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    Attributes
    ----------
    variables_categorical_:
        The group of categorical input variables that will be used for prediction.

    variables_numerical_:
        The group of numerical input variables that will be used for prediction.

    pipeline_:
        A Sickit-learn Pipeline with a dicretiser and/or encoder. Used to determine the
        mean target value per category or bin, per variable.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn the mean target value per category or per bin, for each variable.
    predict:
        Predict using the average of the target mean value across variables.

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
        # boolean value can be interpreted as an integer
        if not isinstance(bins, int) or isinstance(bins, bool):
            raise ValueError(f"bins must be an integer. Got {bins} instead.")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "strategy takes only values equal_width or equal_frequency. Got "
                f"{strategy} instead."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.bins = bins
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the mean target value per category or bin.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas series of shape = [n_samples,]
            The target variable.
        """
        # check if 'X' is a dataframe
        X = _is_dataframe(X)

        # Check X and y for consistent length
        # X, y = check_X_y(X, y, dtype=None)
        # X = pd.DataFrame(X)
        # y = pd.DataFrame(y)
        # identify categorical and numerical variables
        (
            self.variables_categorical_,
            self.variables_numerical_,
        ) = _find_categorical_and_numerical_variables(X, self.variables)

        # check for missing values
        _check_contains_na(X, self.variables_numerical_)
        _check_contains_na(X, self.variables_categorical_)

        # check inf
        _check_contains_inf(X, self.variables_numerical_)
        # pipeline with discretiser and encoder
        if self.variables_categorical_ and self.variables_numerical_:
            self.pipeline_ = self._make_combined_pipeline()

        elif self.variables_categorical_:
            self.pipeline_ = self._make_categorical_pipeline()

        else:
            self.pipeline_ = self._make_numerical_pipeline()

        self.pipeline_.fit(X, y)

        self.n_features_in_ = X.shape[1]

        return self

    def _make_numerical_pipeline(self):
        """
        Create pipeline for a dataframe solely comprised of numerical variables
        using a discretiser and an encoder.
        """
        encoder = MeanEncoder(variables=self.variables_numerical_, errors="raise")

        pipeline = Pipeline(
            [
                ("discretiser", self._make_discretiser()),
                ("encoder", encoder),
            ]
        )

        return pipeline

    def _make_categorical_pipeline(self):
        """
        Instantiate the target mean encoder. Used when all variables are categorical.
        """

        pipeline = MeanEncoder(variables=self.variables_categorical_, errors="raise")

        return pipeline

    def _make_combined_pipeline(self):

        encoder_num = MeanEncoder(variables=self.variables_numerical_, errors="raise")
        encoder_cat = MeanEncoder(variables=self.variables_categorical_, errors="raise")

        pipeline = Pipeline(
            [
                ("discretiser", self._make_discretiser()),
                ("encoder_num", encoder_num),
                ("encoder_cat", encoder_cat),
            ]
        )

        return pipeline

    def _make_discretiser(self):
        """
        Instantiate the EqualWidthDiscretiser or EqualFrequencyDiscretiser.
        """
        if self.strategy == "equal_width":
            discretiser = EqualWidthDiscretiser(
                bins=self.bins, variables=self.variables_numerical_, return_object=True
            )
        else:
            discretiser = EqualFrequencyDiscretiser(
                q=self.bins, variables=self.variables_numerical_, return_object=True
            )

        return discretiser

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the average of the target mean value across variables.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Return
        -------
        y_pred: pandas series of shape = [n_samples, ]
            The mean target value per observation.
        """
        # check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check input data contains same number of columns as df used to fit
        _check_input_matches_training_df(X, self.n_features_in_)

        # transform dataframe
        X_tr = self.pipeline_.transform(X)

        # calculate the average for each observation
        predictions = X_tr[
            self.variables_numerical_ + self.variables_categorical_
        ].mean(axis=1).to_numpy()

        return predictions
