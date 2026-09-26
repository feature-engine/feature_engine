# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

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
from feature_engine.dataframe_checks import _check_X_matches_training_df, check_X
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
class MeanImputer(BaseImputer):
    """
    The MeanImputer() replaces missing data by the mean or median value of the
    variable. It works only with numerical variables.

    You can pass a list of variables to impute. Alternatively, the
    MeanImputer() will automatically select all variables of type numeric in the
    training set.

    More details in the :ref:`User Guide <mean_imputer>`.

    Parameters
    ----------
    imputation_method: str, default='median'
        Desired method of imputation. Can take 'mean' or 'median'.

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
        Learn the mean or median values.

    {fit_transform}

    {transform}

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import MeanImputer
    >>> X = pd.DataFrame(dict(
    >>>        x1 = [np.nan,1,1,0,np.nan],
    >>>        x2 = ["a", np.nan, "b", np.nan, "a"],
    >>>        ))
    >>> mmi = MeanImputer(imputation_method='median')
    >>> mmi.fit(X)
    >>> mmi.transform(X)
        x1   x2
    0  1.0    a
    1  1.0  NaN
    2  1.0    b
    3  0.0  NaN
    4  1.0    a
    """

    def __init__(
        self,
        imputation_method: str = "median",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:

        if imputation_method not in ["median", "mean"]:
            raise ValueError("imputation_method takes only values 'median' or 'mean'")

        self.imputation_method = imputation_method
        self.variables = _check_variables_input_value(variables)

        _check_return_empty_is_bool(return_empty)
        self.return_empty = return_empty

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the mean or median values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas series or None, default=None
            y is not needed in this imputation. You can pass None or y.
        """

        # fit only reads X; avoid copying the full input dataframe.
        X = check_X(X, copy=False)

        # find or check for numerical variables
        if self.variables is None:
            variables_ = find_numerical_variables(X, self.return_empty)
        else:
            variables_ = check_numerical_variables(X, self.variables)

        # find imputation parameters: mean or median
        selected = X[variables_]
        if self.imputation_method == "mean":
            imputer_dict_ = selected.mean().to_dict()

        elif self.imputation_method == "median":
            if len(variables_) and all(
                dtype == np.dtype("float64") for dtype in selected.dtypes
            ):
                # NumPy may partition this owned buffer without a second copy.
                values = selected.to_numpy(copy=True)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "All-NaN slice encountered", RuntimeWarning
                    )
                    warnings.filterwarnings(
                        "ignore", "Mean of empty slice", RuntimeWarning
                    )
                    medians = np.nanmedian(values, axis=0, overwrite_input=True)
                imputer_dict_ = dict(zip(variables_, medians.tolist()))
            else:
                imputer_dict_ = selected.median().to_dict()

        self.variables_ = variables_
        self.imputer_dict_ = imputer_dict_
        self._get_feature_names_in(X)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # The column selection creates the output frame; reading input needs no copy.
        check_is_fitted(self)
        X = check_X(X, copy=False)
        _check_X_matches_training_df(X, self.n_features_in_)
        return X[self.feature_names_in_]


# TODO: remove in version 2.1.0
class MeanMedianImputer(MeanImputer):
    def __init__(
        self,
        imputation_method: str = "median",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:
        warnings.warn(
            "MeanMedianImputer was deprecated in favour of MeanImputer in version "
            "2.0.0 and will be removed in version 2.1.0. To silence this warning, "
            "use MeanImputer instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            imputation_method=imputation_method,
            variables=variables,
            return_empty=return_empty,
        )
