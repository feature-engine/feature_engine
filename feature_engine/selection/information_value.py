from typing import Dict, List, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries

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
from feature_engine.encoding._helper_functions import TARGET_NAME, add_target_to_X
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

    The WoE is not defined for categories or intervals with no positive or no
    negative cases. For those, the transformer replaces the zero count by 0.5 to
    calculate the WoE and the IV.

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

    With polars

    >>> import polars as pl
    >>> from feature_engine.selection import SelectByInformationValue
    >>> X = pl.DataFrame(dict(x1 = [1,1,1,1,1,1],
    >>>                     x2 = [3,2,2,3,3,2],
    >>>                     x3 = ["a","b","c","a","c","b"]))
    >>> y = pl.Series([1,1,1,0,0,0])
    >>> iv = SelectByInformationValue()
    >>> iv.fit_transform(X, y)
    shape: (6, 1)
    ┌─────┐
    │ x2  │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    │ 2   │
    │ 2   │
    │ 3   │
    │ 3   │
    │ 2   │
    └─────┘
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

        if not isinstance(strategy, str) or strategy not in [
            "equal_width",
            "equal_frequency",
        ]:
            raise ValueError(
                "strategy takes only values 'equal_width' or 'equal_frequency'. "
                f"Got {strategy} instead."
            )

        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"threshold must be an integer or a float. Got {threshold} instead."
            )

        super().__init__(confirm_variables)
        self.variables = _check_variables_input_value(variables)
        self.bins = bins
        self.strategy = strategy
        self.threshold = threshold

    def fit(self, X: IntoDataFrame, y: IntoSeries):
        """
        Learn the information value. Find features with IV above the threshold.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: series of shape = [n_samples, ]
            Target, must be binary.
        """
        X, y = self._check_fit_input(X, y)

        self.variables_ = _select_all_variables(
            X, self.variables, self.confirm_variables, exclude_datetime=True
        )

        # variables_ has no datetime variables left, and skipping the datetime
        # check (which parses the values) doesn't change the numerical variables.
        _, variables_numerical = find_categorical_and_numerical_variables(
            X, self.variables_, exclude_datetime=False
        )

        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, variables_numerical)

        self._get_feature_names_in(X)

        if len(variables_numerical) > 0:
            X = self._make_discretiser(variables_numerical).fit_transform(X)

        self.information_values_ = self._calculate_information_values(X, y)

        self.features_to_drop_ = [
            f
            for f in self.information_values_.keys()
            if self.information_values_[f] < self.threshold
        ]

        return self

    def _calculate_information_values(
        self, X: IntoDataFrame, y: IntoSeries
    ) -> Dict[Union[str, int], float]:
        """
        Return the IV of each variable. Zero counts are replaced by 0.5, as in the
        WoEEncoder.
        """
        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            y_arr = y.to_numpy(dtype=float)
            total_pos = y_arr.sum()
            total_neg = len(y_arr) - total_pos
            information_values = {}
            for var in self.variables_:
                codes, _ = X[var].factorize()
                pos = np.bincount(codes, weights=y_arr)
                neg = np.bincount(codes) - pos
                pos = np.where(pos == 0, 0.5, pos) / total_pos
                neg = np.where(neg == 0, 0.5, neg) / total_neg
                information_values[var] = float(np.sum((pos - neg) * np.log(pos / neg)))
            return information_values

        nw_Xy = add_target_to_X(nw.from_native(X, eager_only=True), y)
        total_pos = nw_Xy[TARGET_NAME].sum()
        total_neg = len(nw_Xy) - total_pos

        pos, neg = nw.col("__pos__"), nw.col("__n__") - nw.col("__pos__")
        pos = nw.when(pos == 0).then(0.5).otherwise(pos) / total_pos
        neg = nw.when(neg == 0).then(0.5).otherwise(neg) / total_neg
        iv = ((pos - neg) * (pos / neg).log()).sum().alias("__iv__")

        # a single lazy query lets polars compute the variables in parallel.
        lazy_Xy = nw_Xy.lazy()
        information_values = nw.concat(
            [
                lazy_Xy.group_by(var)
                .agg(
                    nw.col(TARGET_NAME).sum().alias("__pos__"),
                    nw.len().alias("__n__"),
                )
                .select(iv)
                for var in self.variables_
            ],
            how="vertical",
        ).collect()
        return dict(zip(self.variables_, information_values["__iv__"].to_list()))

    def _make_discretiser(self, variables):
        """
        Instantiate the EqualWidthDiscretiser or EqualFrequencyDiscretiser.
        """
        # the IV only needs the interval of each value, so integer codes, which are
        # faster to group than the interval boundaries, are enough.
        if self.strategy == "equal_width":
            discretiser = EqualWidthDiscretiser(bins=self.bins, variables=variables)
        else:
            discretiser = EqualFrequencyDiscretiser(q=self.bins, variables=variables)

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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
