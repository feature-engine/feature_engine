# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import warnings
from typing import Dict, List, Optional, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._base_transformers.mixins import FitFromDictMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.tags import _return_tags


@Substitution(
    variables=_variables_numerical_docstring,
    return_empty=_return_empty_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class LogTransformer(BaseNumericalTransformer, FitFromDictMixin):
    """
    The LogTransformer() applies the natural logarithm or the base 10 logarithm to
    numerical variables, optionally after adding a constant C, i.e., log(x + C).

    By default, C=0, so LogTransformer() only works with positive values.
    If a variable contains a zero or a negative value, the transformer raises
    an error. Note that the default value of C will change from 0 to "auto"
    in version 2.1.0.

    To transform variables that contain zero or negative values, pass a non-zero
    C: either an explicit constant, "auto" to let the transformer determine a
    shift per variable, or a dictionary mapping each variable to its own constant.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    More details in the :ref:`User Guide <log_transformer>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    base: string, default='e'
        Indicates if the natural or base 10 logarithm should be applied. Can take
        values 'e' or '10'.

    C: "auto", int, float or dict, default=0
        The constant C to add to the variable before the logarithm, i.e., log(x + C).

        - If 0 (the default), no constant is added and the variable must be
          strictly positive.
        - If int or float, then log(x + C).
        - If "auto", then C = abs(min(x)) + 1.
        - If dict, dictionary mapping the constant C to apply to each variable.

        Note, when C is a dictionary, the parameter `variables` is ignored, because
        the variables to transform are taken from the dictionary keys.

    Attributes
    ----------
    {variables_}

    C_:
        The constant C added to each variable. Equal to `C`, unless `C = "auto"`, in
        which case it is a dictionary with C = abs(min(variable)) + 1. For strictly
        positive variables, C = 0.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {inverse_transform}

    transform:
        Transform the variables using the logarithm.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.transformation import LogTransformer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100)))
    >>> lt = LogTransformer()
    >>> lt.fit(X)
    >>> X = lt.transform(X)
    >>> X.head()
            x
    0  0.496714
    1 -0.138264
    2  0.647689
    3  1.523030
    4 -0.234153

    With polars:

    >>> import numpy as np
    >>> import polars as pl
    >>> from feature_engine.transformation import LogTransformer
    >>> np.random.seed(42)
    >>> X = pl.DataFrame({"x": list(np.random.lognormal(size=6))})
    >>> lt = LogTransformer()
    >>> lt.fit(X)
    >>> lt.transform(X)
    shape: (6, 1)
    ┌───────────┐
    │ x         │
    │ ---       │
    │ f64       │
    ╞═══════════╡
    │ 0.496714  │
    │ -0.138264 │
    │ 0.647689  │
    │ 1.52303   │
    │ -0.234153 │
    │ -0.234137 │
    └───────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        base: str = "e",
        C: Union[int, float, str, Dict[Union[str, int], Union[float, int]]] = 0,
    ) -> None:

        if base not in ["e", "10"]:
            raise ValueError(
                f"base can take only '10' or 'e' as values. Got {base} instead."
            )

        if not isinstance(C, (int, float, dict)) and C != "auto":
            raise ValueError(
                "C can take only 'auto', integers, floats or dictionaries. "
                f"Got {C} instead."
            )

        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.base = base
        self.C = C

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learn the constant C to add to the variable before the logarithm
        transformation, if C="auto". Otherwise, this transformer does not learn
        parameters.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        if isinstance(self.C, dict):
            X, variables_ = super()._fit_from_dict(X, self.C)
        else:
            X, variables_ = self._fit_setup(X)

        values = nw.from_native(X, eager_only=True).select(variables_).to_numpy()
        values = values.astype(float)

        C_ = self.C

        # 0 for strictly positive variables, abs(min) + 1 (shift to positive)
        # otherwise.
        if self.C == "auto":
            mins = values.min(axis=0)
            c_values = np.where(mins > 0, 0, np.abs(mins) + 1)
            C_ = dict(zip(variables_, c_values.tolist()))

        # C=0 is the original LogTransformer contract: no constant is added,
        # so fail fast at fit time exactly as before this class supported C.
        if C_ == 0 and np.any(values <= 0):
            raise ValueError(
                "Some variables contain zero or negative values, can't apply log"
            )

        self.variables_ = variables_
        self.C_ = C_
        self._get_feature_names_in(X)

        return self

    def _c_as_array(self) -> Union[int, float, np.ndarray]:
        """Broadcastable form of C_: a plain scalar, or a numpy array ordered
        to line up column-wise with self.variables_ when C_ is a dict."""
        if isinstance(self.C_, dict):
            return np.array([self.C_[var] for var in self.variables_], dtype=float)
        return self.C_  # type: ignore[return-value]

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Transform the variables with the logarithm of x plus the constant C.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        if self.C_ == 0:
            error_msg = (
                "Some variables contain zero or negative values, can't apply log"
            )
        else:
            error_msg = (
                "Some variables contain zero or negative values after adding"
                + " constant C, can't apply log."
            )

        nw_X = nw.from_native(X, eager_only=True)
        values = nw_X.select(self.variables_).to_numpy().astype(float)
        shifted = values + self._c_as_array()

        if np.any(shifted <= 0):
            raise ValueError(error_msg)

        # transform
        if self.base == "e":
            result = np.log(shifted)
        else:
            result = np.log10(shifted)

        new_series = [
            nw.new_series(var, result[:, i], backend=nw_X.implementation)
            for i, var in enumerate(self.variables_)
        ]
        X = nw_X.with_columns(*new_series).to_native()

        return X

    def inverse_transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_tr: dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        nw_X = nw.from_native(X, eager_only=True)
        values = nw_X.select(self.variables_).to_numpy().astype(float)
        c_arr = self._c_as_array()

        # inverse_transform
        if self.base == "e":
            result = np.exp(values) - c_arr
        else:
            result = 10**values - c_arr

        new_series = [
            nw.new_series(var, result[:, i], backend=nw_X.implementation)
            for i, var in enumerate(self.variables_)
        ]
        X = nw_X.with_columns(*new_series).to_native()

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # =======  this tests fail because the transformers throw an error
        # when the values are 0 and C=0 (the default). Nothing to do with the
        # test itself but mostly with the data created and used in the test
        msg = (
            "transformers raise errors when data contains zeroes, thus this check fails"
        )
        tags_dict["_xfail_checks"]["check_estimators_dtypes"] = msg
        tags_dict["_xfail_checks"]["check_estimators_fit_returns_self"] = msg
        tags_dict["_xfail_checks"]["check_pipeline_consistency"] = msg
        tags_dict["_xfail_checks"]["check_estimators_overwrite_params"] = msg
        tags_dict["_xfail_checks"]["check_estimators_pickle"] = msg
        tags_dict["_xfail_checks"]["check_transformer_general"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags


# TODO: remove in version 2.1.0
class LogCpTransformer(LogTransformer):
    """
    LogCpTransformer() applies the transformation log(x + C), where x is the
    variable to transform and C is a positive constant.

    .. note::
        `LogCpTransformer` is consolidated into `LogTransformer` and deprecated
        in version 2.0.0. It will be removed in version 2.1.0. New code should prefer
        ``LogTransformer(C="auto")``, which reproduces `LogCpTransformer`'s
        default behavior exactly.

    See :class:`LogTransformer` for the full parameter and attribute reference.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.transformation import LogCpTransformer
    >>> X = pd.DataFrame(dict(
    >>>    vara=[0, 1, 2, 3],
    >>>    varb=[5, 5, 6, 7],
    >>>    varc=[-2, -1, 0, 4],
    >>>    vard=[-3, -2, -1, -5],
    >>>    vare=["a", "b", "c", "d"]))
    >>> lct = LogCpTransformer()
    >>> lct.fit(X)
    >>> X = lct.transform(X)
    >>> X
           vara      varb      varc      vard vare
    0  0.000000  1.609438  0.000000  1.098612    a
    1  0.693147  1.609438  0.693147  1.386294    b
    2  1.098612  1.791759  1.098612  1.609438    c
    3  1.386294  1.945910  1.945910  0.000000    d
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        base: str = "e",
        C: Union[int, float, str, Dict[Union[str, int], Union[float, int]]] = "auto",
    ) -> None:
        super().__init__(
            variables=variables, return_empty=return_empty, base=base, C=C
        )
        warnings.warn(
            "LogCpTransformer was deprecated in version 2.0.0 in favour of "
            "LogTransformer and will be removed in version 2.1.0. "
            'Use LogTransformer(C="auto") instead.',
            FutureWarning,
        )

    def _more_tags(self):
        # LogCpTransformer's default ("auto") always finds a valid shift, so it
        # doesn't hit the zero-value errors LogTransformer's C=0 default does.
        # Restore the un-xfailed tags rather than inheriting LogTransformer's.
        return _return_tags()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
