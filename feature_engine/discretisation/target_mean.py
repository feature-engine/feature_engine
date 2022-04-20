import warnings
from typing import Dict, List, Optional, Union

import pandas as pd

from feature_engine.discretisation.base_discretiser import BaseDiscretiser
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring
)
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.discretisation import (
    ArbitraryDiscretiser,
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser
)
from feature_engine.encoding import MeanEncoder
from feature_engine.tags import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)

from sklearn.pipeline import Pipeline

@Substitution(
    return_objects=BaseDiscretiser._return_object_docstring,
    return_boundaries=BaseDiscretiser._return_boundaries_docstring,
    binner_dict_=BaseDiscretiser._binner_dict_docstring,
    transform=BaseDiscretiser._transform_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class TargetMeanDiscretiser(BaseDiscretiser):
    """

    Parameters
    ----------
    binning_dict: dict
        The dictionary with the variable to interval limits pairs.

    {return_object}

    {return_boundaries}

    errors: string, default='ignore'
        Indicates what to do when a value is outside the limits indicated in the
        'binning_dict'. If 'raise', the transformation will raise an error.
        If 'ignore', values outside the limits are returned as NaN
        and a warning will be raised instead.

    Attributes
    ----------
    {variables_}

    {binner_dict_}



    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    See Also
    --------
    pandas.cut
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        bins: int = 10,
        strategy: str = "equal_frequency",
        errors: str = "ignore",
    ) -> None:

        if not isinstance(bins, int):
            raise ValueError(
                f"bins must be an integer. Got {bins} instead."
            )
        if strategy not in ("equal_frequency", "equal_width"):
            raise ValueError(
                "strategy must equal 'equal_frequency' or 'equal_width'. "
                f"Got {strategy} instead."
            )

        if errors not in ("ignore", "raise"):
            raise ValueError(
                "errors only takes values 'ignore' and 'raise. "
                f"Got {errors} instead."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.bins = bins
        self.strategy = strategy
        self.errors = errors

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the boundaries of the selected dicretiser's intervals / bins
        for the chosen numerical variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y : pandas series of shape = [n_samples,]
            y is not needed in this discretiser. You can pass y or None.
        """
        # check if 'X' is a dataframe
        X = check_X(X)

        #  identify numerical variables
        self.variables_numerical_ = _find_or_check_numerical_variables(
            X, self.variables
        )

        # create dataframe to use for target values.
        self.X_target_ = X[self.variables_numerical_].copy()

        # check for missing values
        _check_contains_na(X, self.variables_numerical_)

        # check for inf
        _check_contains_inf(X, self.variables_numerical_)

        # discretise
        self._discretiser = self._make_discretiser()
        self._discretiser.fit(X)

        # store input features
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace original values by the average of the target mean value per bin
        for each of the variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_enc: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the means of the selected numerical variables.

        """
        # check that fit method has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # check that input data contain number of columns as the fitted df
        _check_X_matches_training_df(X, self.n_features_in_)

        # check for missing values
        _check_contains_na(X, self.variables_numerical_)

        # check for infinite values
        _check_contains_inf(X, self.variables_numerical_)

        # discretise
        X_disc = self._discretiser.transform(X)

        # encode
        X_enc = self._encode_X(X_disc)

        return X_enc

    def _make_discretiser(self):
        """
        Instantiate the EqualFrequencyDiscretiser or EqualWidthDiscretiser.
        """
        if self.strategy == "equal_frequency":
            discretiser = EqualFrequencyDiscretiser(
                q=self.bins,
                variables=self.variables_numerical_,
                return_boundaries=True,
            )
        else:
            discretiser = EqualWidthDiscretiser(
                bins=self.bins,
                variables=self.variables_numerical_,
                return_boundaries=True
            )

        return discretiser

    def _encode_X(self, X):
        """
        Calculate the mean of each bin using the initial values (prior to
        discretisation) for each selected variable. Replace the discrete value
        (bin) with the corresponding mean.
        """
        X_enc = X.copy()
        X_enc[self.variables_numerical_] = X_enc[self.variables_numerical_].astype(str)

        for variable in self.variables_numerical_:
            encoder = MeanEncoder(variables=variable)
            encoder.fit(X_enc, self.X_target_[variable])
            X_enc = encoder.transform(X_enc)

        return X_enc