# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser
)
from feature_engine.encoding import MeanEncoder
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
)


class TargetMeanPredictor(BaseEstimator):
    """

    Parameters
    ----------
    variables: list, default=None
        The list of input variables. If None, the estimator will evaluate will use all
        variables as input fetures.

    bins: int, default=5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default='equal_width'
        Whether the bins should of equal width ('equal_width') or equal frequency ('equal_frequency').

    Attributes
    ----------


    Methods
    -------
    fit:

    predict:

    Notes
    -----


    See Also
    --------
    feature_engine.encoding.MeanEncoder
    feature_engine.discretisation.EqualWidthDiscretiser
    feature_engine.discretisation.EqualFrequencyDiscretiser

    References
    ----------


    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        bins: int = 5,
        strategy: str = "equal-width",
    ):

        if not isinstance(bins, int):
            raise TypeError(f"Got {bins} bins instead of an integer.")

        if strategy not in ("equal-width", "equal-distance"):
            raise ValueError(
                "strategy must be 'equal-width' or 'equal-distance'."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.bins = bins
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit predictor per variables.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas series of shape = [n_samples,]
            The target variable.
        """

        # check variables
        self.variables_ = _find_all_variables(X, self.variables)

        # identify categorical and numerical variables
        self.variables_categorical_ = list(X[self.variables_].select_dtypes(include="object").columns)
        self.variables_numerical_ = list(X[self.variables_].select_dtypes(include="number").columns)

        # encode categorical variables and discretise numerical variables
        if self.variables_categorical_ and self.variables_numerical_:
            _pipeline = self._make_combined_pipeline()

        elif self.variables_categorical_:
            _pipeline = self._make_categorical_pipeline()

        else:
            _pipeline = self._make_numerical_pipeline()

        _pipeline.fit(X, y)

        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X: pd.DataFrame, variable_name: str) -> pd.Series:
        """

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, ]
            The input series which must have the same name as one of the features in the
            dataframe that was used to fit the predictor.

        variable_name: str
            The variable the method should transform and return.

        Return
        -------
        X_prediction: pandas series of shape = [n_samples, ]
            Values are the mean values associated with the corresponding encoded or discretised bin

        """
        # NOTES:
        # - X needs to be a dataframe to be compatible w/ the BaseEncoder()
        # - X needs to match the shape of the dataframe used in fit()


        # check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        _is_dataframe(X)

        # Check input data contains same number of columns as df used to fit
        _check_input_matches_training_df(X, self.n_features_in_)




    def _make_numerical_pipeline(self):
        """
        Create pipeline for a dataframe solely comprised of numerical variables 
        using a discretiser and encoder.
        """"

        encoder = MeanEncoder(variables=self.variables_numerical_, errors="raise")

        _pipeline_numerical = Pipeline(
            [
                ("discretisation": self._make_disretiser()),
                ("encoder": encoder),
            ]
        )

        return _pipeline_numerical

    def _make_categorical_pipeline(self):
        """
        Instantiate the encoder for a dataframe solely comprised of categorical variables.
        """

        return MeanEncoder(variables=self.variables_categorical_, errors="raise")

    def _make_combined_pipeline(self):

        encoder_num = MeanEncoder(variables=self.variables_numerical_, errors="raise")
        encoder_cat = MeanEncoder(variables=self.variables_categorical_, errors="raise")

        _pipeline_combined = Pipeline(
            [
                ("discretisation": self._make_discretiser()),
                ("encoder_num": encoder_num),
                ("encoder_cat": encoder_cat),
            ]
        )

        return _pipeline_combined

    def _make_discretiser(self):
        """


        """
        if self.strategy == "equal-width":
            discretiser = EqualWidthDiscretiser(
                bins=self.bins, variables=self.variables_numerical_, return_object=True
            )
        else:
            discretiser = EqualFrequencyDiscretiser(
                q=self.bins, variables=self.variables_numerical_, return_object=True
        )

        return discretiser

