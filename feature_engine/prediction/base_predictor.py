# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator

from feature_engine.dataframe_checks import (
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser
)
from feature_engine.encoding import MeanEncoder

class BaseTargetMeanPredictor(BaseEstimator):
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
        Whether the bins should of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    Attributes
    ----------


    Methods
    -------
    fit:

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
        strategy: str = "equal_width",
    ):

        if not isinstance(bins, int):
            raise TypeError(f"Got {bins} bins instead of an integer.")

        if strategy not in ("equal_width", "equal_distance"):
            raise ValueError(
                "strategy must be 'equal_width' or 'equal_distance'."
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
        # check if 'X' is a dataframe
        _is_dataframe(X)

        # check variables
        self.variables_ = _find_all_variables(X, self.variables)

        # identify categorical and numerical variables
        self.variables_categorical_ = list(
            X[self.variables_].select_dtypes(include="object").columns
        )
        self.variables_numerical_ = list(
            X[self.variables_].select_dtypes(include="number").columns
        )

        # encode categorical variables and discretise numerical variables
        if self.variables_categorical_ and self.variables_numerical_:
            pipeline = self._make_combined_pipeline()

        elif self.variables_categorical_:
            pipeline = self._make_categorical_pipeline()

        else:
            pipeline = self._make_numerical_pipeline()

        self.pipeline = pipeline
        self.pipeline.fit(X, y)

        self.n_features_in_ = X.shape[1]

        return self

    def _make_numerical_pipeline(self):
        """
        Create pipeline for a dataframe solely comprised of numerical variables
        using a discretiser and encoder.
        """
        encoder = MeanEncoder(variables=self.variables_numerical_, errors="raise")

        _pipeline_numerical = Pipeline(
            [
                ("discretisation", self._make_discretiser()),
                ("encoder", encoder),
            ]
        )

        return _pipeline_numerical

    def _make_categorical_pipeline(self):
        """
        Instantiate the encoder for a dataframe solely comprised of categorical
        variables.
        """

        return MeanEncoder(
            variables=self.variables_categorical_, errors="raise"
        )

    def _make_combined_pipeline(self):

        encoder_num = MeanEncoder(variables=self.variables_numerical_, errors="raise")
        encoder_cat = MeanEncoder(variables=self.variables_categorical_, errors="raise")

        _pipeline_combined = Pipeline(
            [
                ("discretisation", self._make_discretiser()),
                ("encoder_num", encoder_num),
                ("encoder_cat", encoder_cat),
            ]
        )

        return _pipeline_combined

    def _make_discretiser(self):
        """
        Instantiate either EqualWidthDiscretiser or EqualFrequencyDiscretiser.
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