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