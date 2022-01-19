# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import warnings
from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _find_all_variables,
    _find_or_check_categorical_variables,
    _check_input_parameter_variables,
)