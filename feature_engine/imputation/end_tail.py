# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Dict, List, Optional, Union

from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _imputer_dict_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _variables_numerical_docstring, _return_empty_docstring
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_imputers_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.dataframe_helpers import aggregate_columns
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    return_empty=_return_empty_docstring,
    imputer_dict_=_imputer_dict_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=_transform_imputers_docstring,
    fit_transform=_fit_transform_docstring,
)
class EndTailImputer(BaseImputer):
    """
    The EndTailImputer() replaces missing data by a value at either tail of the
    distribution. It works only with numerical variables.

    You can indicate the variables to impute in a list. Alternatively, the
    EndTailImputer() will automatically select all numerical variables.

    The imputer first calculates the values at the end of the distribution for each
    variable (fit). The values at the end of the distribution are determined using
    the Gaussian limits, the IQR proximity rule limits, or a factor of the maximum
    value:

    Gaussian limits:
        - right tail: mean + 3*std
        - left tail: mean - 3*std

    IQR limits:
        - right tail: 75th quantile + 3*IQR
        - left tail:  25th quantile - 3*IQR

    where IQR is the inter-quartile range = 75th quantile - 25th quantile

    Maximum value:
        - right tail: max * 3
        - left tail: not applicable

    You can change the factor that multiplies the std, IQR or the maximum value
    using the parameter `fold` (we used `fold=3` in the examples above).

    The imputer then replaces the missing data with the estimated values (transform).

    More details in the :ref:`User Guide <end_tail_imputer>`.

    Parameters
    ----------
    imputation_method: str, default='gaussian'
        Method to be used to find the replacement values. Can take 'gaussian',
        'iqr' or 'max'.

        **'gaussian'**: the imputer will use the Gaussian limits to find the values
        to replace missing data.

        **'iqr'**: the imputer will use the IQR limits to find the values to replace
        missing data.

        **'max'**: the imputer will use the maximum values to replace missing data. Note
        that if 'max' is passed, the parameter 'tail' is ignored.

    tail: str, default='right'
        Indicates if the values to replace missing data should be selected from the
        right or left tail of the variable distribution. Can take values 'left' or
        'right'.

    fold: int, default=3
        Factor to multiply the std, the IQR or the Max values. Recommended values
        are 2 or 3 for Gaussian, or 1.5 or 3 for IQR.

    {variables}

    {return_empty}

    Attributes
    ----------
    {imputer_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn values to replace missing data.

    {fit_transform}

    {transform}

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import EndTailImputer
    >>> X = pd.DataFrame(dict(x1 = [np.nan,0.5, 0.5, 0,np.nan]))
    >>> eti = EndTailImputer(imputation_method='gaussian', tail='right', fold=3)
    >>> eti.fit(X)
    >>> eti.transform(X)
             x1
    0  1.199359
    1  0.500000
    2  0.500000
    3  0.000000
    4  1.199359

    With polars:

    >>> import polars as pl
    >>> from feature_engine.imputation import EndTailImputer
    >>> X = pl.DataFrame({"x1": [None, 0.5, 0.5, 0.0, None]})
    >>> eti = EndTailImputer(imputation_method='gaussian', tail='right', fold=3)
    >>> eti.fit(X)
    >>> eti.transform(X)
    shape: (5, 1)
    ┌──────────┐
    │ x1       │
    │ ---      │
    │ f64      │
    ╞══════════╡
    │ 1.199359 │
    │ 0.5      │
    │ 0.5      │
    │ 0.0      │
    │ 1.199359 │
    └──────────┘
    """

    def __init__(
        self,
        imputation_method: str = "gaussian",
        tail: str = "right",
        fold: int = 3,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:

        if not isinstance(imputation_method, str) or imputation_method not in [
            "gaussian",
            "iqr",
            "max",
        ]:
            raise ValueError(
                "imputation_method takes only values 'gaussian', 'iqr' or 'max'. "
                f"Got {imputation_method} instead."
            )

        if not isinstance(tail, str) or tail not in ["right", "left"]:
            raise ValueError(
                f"tail takes only values 'right' or 'left'. Got {tail} instead."
            )

        if not isinstance(fold, (int, float)) or isinstance(fold, bool) or fold <= 0:
            raise ValueError(f"fold takes only positive numbers. Got {fold} instead.")

        self.imputation_method = imputation_method
        self.tail = tail
        self.fold = fold
        self.variables = _check_variables_input_value(variables)

        _check_return_empty_is_bool(return_empty)
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learn the values at the end of the variable distribution.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.
        """
        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        if self.variables is None:
            variables_ = find_numerical_variables(X, self.return_empty)
        else:
            variables_ = check_numerical_variables(X, self.variables)

        imputer_dict_ = self._end_values(X, variables_)

        self.variables_ = variables_
        self.imputer_dict_ = imputer_dict_
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def _end_values(self, X, variables: List[Union[str, int]]) -> Dict:
        """Compute the end-of-distribution replacement value of each variable, per
        `imputation_method` and `tail`."""
        if self.imputation_method == "max":
            maxs = aggregate_columns(X, variables, "max")
            return {var: maxs[var] * self.fold for var in variables}

        if self.imputation_method == "gaussian":
            means = aggregate_columns(X, variables, "mean")
            stds = aggregate_columns(X, variables, "std")
            if self.tail == "right":
                return {var: means[var] + self.fold * stds[var] for var in variables}
            return {var: means[var] - self.fold * stds[var] for var in variables}

        # imputation_method == "iqr"
        q1 = aggregate_columns(X, variables, "quantile", 0.25)
        q3 = aggregate_columns(X, variables, "quantile", 0.75)
        if self.tail == "right":
            return {var: q3[var] + self.fold * (q3[var] - q1[var]) for var in variables}
        return {var: q1[var] - self.fold * (q3[var] - q1[var]) for var in variables}
