# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
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


class TargetMeanPredictor(BaseEstimator, ClassifierMixin, RegressorMixin):
    """

    Parameters
    ----------

    bins: int, default=5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    numeric_var_strategy: str, default='equal_width'
        Whether to create the bins for discretization of numerical variables using
        equal width ('equal_width') or equal frequency ('equal_frequency').

    ignore_format: bool, default=False
        Whether the format in which the categorical variables are cast should be
        ignored. If False, the encoder will automatically select variables of type
        object or categorical, or check that the variables entered by the user are of
        type object or categorical. If True, the encoder will select all variables or
        accept all variables entered by the user, including those cast as numeric.

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
        bins: int = 5,
        numeric_var_strategy: str = "equal-width",
        ignore_format: bool = False,
    ):

        if numeric_var_strategy not in ("equal-width", "equal-distance"):
            raise ValueError(
                "numeric_var_strategy must be 'equal-width' or 'equal-distance'."
            )

        if not isinstance(ignore_format, bool):
            raise ValueError("ignore_format takes only booleans True and False")
        
        self.bins = bins
        self.numeric_var_strategy = numeric_var_strategy
        self.ignore_format = ignore_format

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
        # check if dataframe
        _is_dataframe(X)

        # check for NaN values
        _check_contains_na(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if self.regression is True:
            if self.numeric_var_strategy == "equal-width":
                transformer = EqualWidthDiscretiser()
            elif self.numeric_var_strategy == "equal-distance":
                transformer = EqualFrequencyDiscretiser()

            transformer.fit(X, y)
            X = transformer.transform(X)





    def predict(self, X: pd.DataFrame) -> pd.Series:
        """

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The inputs uses to derive the predictions.

        Return
        -------
        y : pandas series of (n_samples,)
            Mean target values.

        """
        pass