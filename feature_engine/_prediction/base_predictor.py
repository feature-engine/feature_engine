from typing import List, Union

import numpy as np
import pandas as pd
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
        Whether the bins should of equal width ('equal_width') or equal frequency
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

        if not isinstance(bins, int):
            raise ValueError(f"bins must be an integer. Got {bins} instead.")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "strategy takes only values 'equal_width' or 'equal_frequency'. "
                f"Got {strategy} instead."
            )

        self.variables = _check_variables_input_value(variables)
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
        X, y = check_X_y(X, y)

        # find categorical and numerical variables
        (
            self.variables_categorical_,
            self.variables_numerical_,
        ) = find_categorical_and_numerical_variables(X, self.variables)

        # check for missing values
        _check_contains_na(X, self.variables_numerical_)
        _check_contains_na(X, self.variables_categorical_)

        # check inf
        _check_contains_inf(X, self.variables_numerical_)

        # Create pipelines
        if self.variables_categorical_ and self.variables_numerical_:
            self._pipeline = self._make_combined_pipeline()

        elif self.variables_categorical_:
            self._pipeline = self._make_categorical_pipeline()

        else:
            self._pipeline = self._make_numerical_pipeline()

        # Train pipeline
        self._pipeline.fit(X, y)

        # Assign attributes (useful to interpret features)
        # Use dict() to make a copy of the dictionary. Otherwise, like in pandas,
        # it is just another view of the same data, mind-blowing.
        if self.variables_categorical_ and self.variables_numerical_:
            self.binner_dict_ = dict(
                self._pipeline.named_steps["discretiser"].binner_dict_
            )
            self.encoder_dict_ = dict(
                self._pipeline.named_steps["encoder_num"].encoder_dict_
            )
            tmp_dict = dict(self._pipeline.named_steps["encoder_cat"].encoder_dict_)
            self.encoder_dict_.update(tmp_dict)

        elif self.variables_categorical_:
            self.binner_dict_ = {}
            self.encoder_dict_ = dict(self._pipeline.encoder_dict_)

        else:
            self.binner_dict_ = dict(
                self._pipeline.named_steps["discretiser"].binner_dict_
            )
            self.encoder_dict_ = dict(
                self._pipeline.named_steps["encoder"].encoder_dict_
            )

        # store input features
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)

        return self

    def _make_numerical_pipeline(self):
        """
        Create pipeline for a dataframe solely comprised of numerical variables
        using a discretiser and an encoder.
        """
        encoder = MeanEncoder(variables=self.variables_numerical_, unseen="raise")

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

        pipeline = MeanEncoder(variables=self.variables_categorical_, unseen="raise")

        return pipeline

    def _make_combined_pipeline(self):

        encoder_num = MeanEncoder(variables=self.variables_numerical_, unseen="raise")
        encoder_cat = MeanEncoder(variables=self.variables_categorical_, unseen="raise")

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
                bins=self.bins,
                variables=self.variables_numerical_,
                return_boundaries=True,
            )
        else:
            discretiser = EqualFrequencyDiscretiser(
                q=self.bins,
                variables=self.variables_numerical_,
                return_boundaries=True,
            )

        return discretiser

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
         Replace original values by the average of the target mean value per bin or
         category in each one of the variables.

         Parameters
         ----------
         X : pandas dataframe of shape = [n_samples, n_features]
             The input samples.

         Return
         -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """
        # check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check input data contains same number of columns as df used to fit
        _check_X_matches_training_df(X, self.n_features_in_)

        # check for missing values
        _check_contains_na(X, self.variables_numerical_)
        _check_contains_na(X, self.variables_categorical_)

        # check inf
        _check_contains_inf(X, self.variables_numerical_)

        # reorder dataframe to match train set
        X = X[self.feature_names_in_]

        # transform dataframe
        X_tr = self._pipeline.transform(X)

        return X_tr

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the average of the target mean value across variables.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Return
        -------
        y_pred: numpy array of shape = (n_samples, )
            The mean target value per observation.
        """
        # transform dataframe
        X_tr = self._transform(X)

        # calculate the average for each observation
        predictions = (
            X_tr[self.variables_numerical_ + self.variables_categorical_]
            .mean(axis=1)
            .to_numpy()
        )

        return predictions

    def _more_tags(self):
        return _return_tags()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
