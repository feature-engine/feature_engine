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
class TargetMeanDiscretiser(BaseeTargetMeanEstimator, BaseDiscretiser):
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
        bins: int = 5,
        strategy: str = "equal_frequency",
        binning_dict: Dict[Union[str, int], List[Union[str, int]]] = None,
        errors: str = "ignore",
    ) -> None:
        # TODO: do we include ArbitraryDiscretiser?
        if strategy not in ("arbitrary", "equal_frequency", "equal_width"):
            raise ValueError(
                "strategy must equal 'arbitrary', 'equal_frequency', 'equal_width'. "
                f"Got {strategy} instead."
            )

        if strategy == "arbitrary" and not isinstance(binning_dict, dict):
            raise ValueError(
                "If 'arbitrary' is the selected strategy, then binning_dict must be a "
                f"dictionary with the interval limits per variable. Got {binning_dict} instead."
            )

        if errors not in ["ignore", "raise"]:
            raise ValueError(
                "errors only takes values 'ignore' and 'raise. "
                f"Got {errors} instead."
            )

        self.variables = variables
        self.bins = bins
        self.strategy = strategy
        self.binning_dict = binning_dict
        self.errors = errors

    def fit(self, X: pd.DataFrame, y:Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.
        y: None
            y is not needed in this transformer. You can pass y or None.
        """
        # check if 'X' is a dataframe
        X = check_X(X)

        # identify numerical variables
        self.variables_numerical_ = _find_or_check_numerical_variables(
            X, self.variables
        )

        # check for missing values
        _check_contains_na(X, self.variables_numerical_)

        # check for inf
        _check_contains_inf(X, self.variables_numerical_)

        # create this attribute for consistency with the rest of the discretisers
        if self.strategy == "arbitrary":
            # check dataframe
            X = super()._fit_from_dict(X, self.binning_dict)
            self.binner_dict_ = self.binning_dict

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the means of the discrete variables.

        """

        # checks if dataset contains na or inf
        X = super().transform(X)

    def _make_discretiser(self):
        """
        Instantiate the ArbirtraryDiscretiser, EqualFrequencyDiscretiser, or
        EqualWidthDiscretiser.
        """
        # TODO: do we include ArbitraryDiscretiser?
        if self.strategy == "arbitrary":
            discretiser = ArbitraryDiscretiser(

            )

        elif self.strategy == "equal_frequency":
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

    def _make_pipeline(self):
        """
        Instantiate target mean encoder and create pipeline of selected
        discretiser and encoder.
        """
        encoder = MeanEncoder(variables=self.variables_numerical_, errors="raise")

        pipeline = Pipeline(
            [
                ("discretiser", self._make_discretiser()),
                ("encoder", encoder),
            ]
        )

        return pipeline