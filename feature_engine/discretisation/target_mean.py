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
from feature_engine.tags import _return_tags


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
    {binner_dict_}

    {variables_}

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
        binning_dict: Dict[Union[str, int], List[Union[str, int]]],
        return_object: bool = False,
        return_boundaries: bool = False,
        errors: str = "ignore",
    ) -> None:

        if not isinstance(binning_dict, dict):
            raise ValueError(
                "binning_dict must be a dictionary with the interval limits per "
                f"variable. Got {binning_dict} instead."
            )

        if errors not in ["ignore", "raise"]:
            raise ValueError(
                "errors only takes values 'ignore' and 'raise. "
                f"Got {errors} instead."
            )

        super().__init__(return_object, return_boundaries)

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
        # check dataframe
        X = super()._fit_from_dict(X, self.binning_dict)

        # create this attribute for consistency with the rest of the discretisers
        self.binner_dict_ = self.binning_dict

    def transform(self,):