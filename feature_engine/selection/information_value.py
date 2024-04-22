from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import _ignore_format_docstring
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.selection._docstring import (
    _features_to_drop_docstring,
    _get_support_docstring,
    _threshold_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_contains_inf, _check_contains_na
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)
from feature_engine.encoding.woe import WoE
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import find_categorical_and_numerical_variables

from .base_selection_functions import _select_all_variables

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    threshold=_threshold_docstring,
    ignore_format=_ignore_format_docstring,
    variables_=_variables_attribute_docstring,
    features_to_drop=_features_to_drop_docstring,
    feature_names_in=_feature_names_in_docstring,
    n_features_in=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    confirm_variables=_confirm_variables_docstring,
    get_support=_get_support_docstring,
)
class SelectByInformationValue(BaseSelector, WoE):
    """
    SelectByInformationValue() selects features based on their information value (IV).
    The IV is calculated as:

     .. math::

       IV = ∑ (fraction of positive cases - fraction of negative cases) * WoE

    where:

    - the fraction of positive cases is the proportion of observations of class 1,
        from the total class 1 observations.
    - the fraction of negative cases is the proportion of observations of class 0,
        from the total class 0 observations.
    - WoE is the weight of the evidence.

    SelectByInformationValue() is only suitable to select features for binary
    classification.

    SelectByInformationValue() can determine the IV for numerical and categorical
    variables. For numerical variables, it first sorts the variables into intervals,
    and then determines the IV.

    You can pass a list of variables to examine. Alternatively, the transformer will
    examine all variables.

    The IV allows you to assess each variable's independent contribution to the target
    variable. The transformer selects those variables whose IV is higher than the
    threshold.

    More details in the :ref:`User Guide <information_value>`.


    Parameters
    ----------
    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset (except datetime).

    bins: int, default = 5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default = 'equal_width'
        Whether the bins should be of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    threshold: float, int, default = 0.2.
        The threshold to drop a feature. If the IV for a feature is < threshold, the
        feature will be dropped.

    {confirm_variables}

    Attributes
    ----------
    {variables_}

    information_values_:
        A dictionary with the information values for each feature.

    {features_to_drop}

    {feature_names_in}

    {n_features_in}

    Methods
    -------
    fit:
        Find features with high information value.

    {fit_transform}

    {get_support}

    transform:
        Remove features with low information value.

    See Also
    --------
    feature_engine.encoding.WoEEncoder
    feature_engine.discretisation.EqualWidthDiscretiser
    feature_engine.discretisation.EqualFrequencyDiscretiser

    References
    ----------
    .. [1] Weight of evidence and information value explained
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    .. [2] WoE and IV for continuous variables
        https://www.listendata.com/2019/08/WOE-IV-Continuous-Dependent.html

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import SelectByInformationValue
    >>> X = pd.DataFrame(dict(x1 = [1,1,1,1,1,1],
    >>>                     x2 = [3,2,2,3,3,2],
    >>>                     x3 = ["a","b","c","a","c","b"]))
    >>> y = pd.Series([1,1,1,0,0,0])
    >>> iv = SelectByInformationValue()
    >>> iv.fit_transform(X, y)
        x2
    0   3
    1   2
    2   2
    3   3
    4   3
    5   2
    """

    def __init__(
        self,
        variables: Variables = None,
        bins: int = 5,
        strategy: str = "equal_width",
        threshold: Union[float, int] = 0.2,
        confirm_variables: bool = False,
    ) -> None:

        if not isinstance(bins, int) or isinstance(bins, int) and bins <= 0:
            raise ValueError(f"bins must be an integer. Got {bins} instead.")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "strategy takes only values 'equal_width' or 'equal_frequency'. "
                f"Got {strategy} instead."
            )

        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"threshold must be a an integer or a float. Got {threshold} "
                "instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.bins = bins
        self.strategy = strategy
        self.threshold = threshold
        self.confirm_variables = confirm_variables

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the information value. Find features with IV above the threshold.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas series of shape = [n_samples, ]
            Target, must be binary.
        """
        # check input dataframe
        X, y = self._check_fit_input(X, y)

        # find categorical and numerical variables
        # find all variables or check those entered are present in the dataframe
        self.variables_ = _select_all_variables(
            X, self.variables, self.confirm_variables, exclude_datetime=True
        )

        _, variables_numerical = find_categorical_and_numerical_variables(
            X, self.variables_
        )

        # check for missing values
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, variables_numerical)

        # get input df features number and name
        self._get_feature_names_in(X)

        # If there are numerical variables, discretize them
        if len(variables_numerical) > 0:
            discretiser = self._make_discretiser(variables_numerical)
            X = discretiser.fit_transform(X)

        self.information_values_ = {}
        for var in self.variables_:
            total_pos, total_neg, woe = self._calculate_woe(X, y, var)
            iv = self._calculate_iv(total_pos, total_neg, woe)
            self.information_values_[var] = iv

        self.features_to_drop_ = [
            f
            for f in self.information_values_.keys()
            if self.information_values_[f] < self.threshold
        ]

        return self

    def _calculate_iv(self, pos, neg, woe):
        return np.sum((pos - neg) * woe)

    def _make_discretiser(self, variables):
        """
        Instantiate the EqualWidthDiscretiser or EqualFrequencyDiscretiser.
        """
        if self.strategy == "equal_width":
            discretiser = EqualWidthDiscretiser(
                bins=self.bins,
                variables=variables,
                return_boundaries=True,
            )
        else:
            discretiser = EqualFrequencyDiscretiser(
                q=self.bins,
                variables=variables,
                return_boundaries=True,
            )

        return discretiser

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "all"
        tags_dict["requires_y"] = True
        tags_dict["binary_only"] = True
        # in the current format, the tests are performed using continuous np.arrays
        # this means that when we encode some of the values, the denominator is 0
        # and this the transformer raises an error, and the test fails.
        # For this reason, most sklearn transformers will fail. And it has nothing to
        # do with the class not being compatible, it is just that the inputs passed
        # are not suitable
        tags_dict["_skip_test"] = True
        return tags_dict
