# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union

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
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
)


class TargetMeanPredictor(BaseEstimator, ClassifierMixin, RegressorMixin):
    """

    Parameters
    ----------
    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.

    bins: int, default=5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default='equal_width'
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
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        bins: int = 5,
        strategy: str = "equal-width",
        ignore_format: bool = False,
    ):

        if not isinstance(bins, int):
            raise TypeError("'bins' only accepts integers.")

        if strategy not in ("equal-width", "equal-distance"):
            raise ValueError(
                "strategy must be 'equal-width' or 'equal-distance'."
            )

        if not isinstance(ignore_format, bool):
            raise ValueError("ignore_format takes only booleans True and False")

        self.variables = _check_input_parameter_variables(variables)
        self.bins = bins
        self.strategy = strategy
        self.ignore_format = ignore_format

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit predictor per variables.

        QUESTION: Does X accept the entire dataframe or just on feature?

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas series of shape = [n_samples,]
            The target variable.
        """
        # check if dataframe
        _is_dataframe(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # check variables
        if self.variables is None:
            self.variables_ = list(X.columns)
        else:
            self.variables_ = _find_all_variables(X, self.variables)

        # identify categorical and numerical variables
        self.variables_categorical_ = list(X.select_dtypes(include="object").columns)
        self.variables_numerical_ = list(X.select_dtypes(include="number").columns)

        # transform categorical variables using the MeanEncoder
        # Should I make this a distinct method?
        self.encoder = MeanEncoder(variables=self.variables_categorical_)
        self.encoder.fit(X, y)

        X_encoded = self.encoder.transform(X)
        X_encoded = X_encoded[self.variables_categorical_]

        # discretise the numerical variables using the EqualWithDiscretiser or EqualDistanceDiscretiser.
        # Should I make this a distinct method?
        if self.strategy == "equal-width":
            self.discretiser = EqualWidthDiscretiser(
                variables=self.variables_numerical_,
                bins=self.bins,
            )
        else:
            self.discretiser = EqualFrequencyDiscretiser(
                variables=self.variables_numerical_,
                bins=self.bins,
            )

        self.discretiser.fit(X, y)
        X_discretised = self.discretiser.transform(X)
        X_discretised = X_discretised[self.variables_numerical_]

        self.disc_mean_dict_ = {}
        temp = pd.concat([X_encoded, X_discretised, y], axis=1)
        temp.columns = list(X_encoded.columns) + list(X_discretised) + ["target"]

        for var in self.variables_numerical_:
            self.disc_mean_dict_[var] = temp.groupby(var)["target"].mean().to_dict()

        # check if df contains na
        _check_contains_na(X, self.variables_)

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

        # check if is pandas series w/ a name that matches
        #
        # if not isinstance(X, pd.Series):
        #     raise TypeError("fit() method only accepts pandas series.")

        if variable_name not in (self.variables_):
            raise ValueError("Series name does not match the dataframe features that were used to "
                             "fit the predictor. The variables that were fitted to the predictor: "
                             f"{self.varialbes_}.")

        # use fitted encoder to return the corresponding mean for each category
        if variable_name in self.variables_categorical_:
            X_prediction = self.encoder.transform(X)

        # transform the numerical to the appropriate discretised bin.
        # replace the bin index w/ the corresponding mean value.
        else:
            X_transformed = self.discretiser.transform(X)
            X_prediction = X_transformed.apply(lambda bin_idx: self.disc_mean_dict_[bin_idx])

        return X_prediction[variable_name]

    def _make_categorical_pipeline(self):

        return MeanEncoder(variables=self.variables_categorical_)

