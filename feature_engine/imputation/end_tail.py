# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _imputer_dict_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_imputers_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
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
    the Gaussian limits, the the IQR proximity rule limits, or a factor of the maximum
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
    """

    def __init__(
        self,
        imputation_method: str = "gaussian",
        tail: str = "right",
        fold: int = 3,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if imputation_method not in ["gaussian", "iqr", "max"]:
            raise ValueError(
                "imputation_method takes only values 'gaussian', 'iqr' or 'max'"
            )

        if tail not in ["right", "left"]:
            raise ValueError("tail takes only values 'right' or 'left'")

        if fold <= 0:
            raise ValueError("fold takes only positive numbers")

        self.imputation_method = imputation_method
        self.tail = tail
        self.fold = fold
        self.variables = _check_variables_input_value(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the values at the end of the variable distribution.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.
        """
        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        # estimate imputation values
        if self.imputation_method == "max":
            self.imputer_dict_ = (X[self.variables_].max() * self.fold).to_dict()

        elif self.imputation_method == "gaussian":
            if self.tail == "right":
                self.imputer_dict_ = (
                    X[self.variables_].mean() + self.fold * X[self.variables_].std()
                ).to_dict()
            elif self.tail == "left":
                self.imputer_dict_ = (
                    X[self.variables_].mean() - self.fold * X[self.variables_].std()
                ).to_dict()

        elif self.imputation_method == "iqr":
            IQR = X[self.variables_].quantile(0.75) - X[self.variables_].quantile(0.25)
            if self.tail == "right":
                self.imputer_dict_ = (
                    X[self.variables_].quantile(0.75) + (IQR * self.fold)
                ).to_dict()
            elif self.tail == "left":
                self.imputer_dict_ = (
                    X[self.variables_].quantile(0.25) - (IQR * self.fold)
                ).to_dict()

        self._get_feature_names_in(X)

        return self
