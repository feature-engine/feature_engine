from typing import List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.selection._docstring import (
    _get_support_docstring,
    _variables_all_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_contains_na, check_X
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags

from .base_selection_functions import _select_all_variables

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    confirm_variables=_confirm_variables_docstring,
    variables=_variables_all_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class DropConstantFeatures(BaseSelector):
    """
    DropConstantFeatures() drops constant and quasi-constant variables from a dataframe.
    Constant variables show the same value in all the observations in the dataset.
    Quasi-constant variables show the same value in almost all the observations in the
    dataset.

    This transformer works with numerical and categorical variables. The user can
    indicate a list of variables to examine. Alternatively, the transformer will
    evaluate all the variables in the dataset.

    The transformer will first identify and store the constant and quasi-constant
    variables. Next, the transformer will drop these variables from a dataframe.

    More details in the :ref:`User Guide <drop_constant>`.

    Parameters
    ----------
    {variables}

    tol: float,int,  default=1
        Threshold to detect constant/quasi-constant features. Variables showing the
        same value in a proportion of observations equal to or greater than tol will
        be considered constant / quasi-constant and dropped. If tol=1, the
        transformer removes constant variables. Else, it will remove quasi-constant
        variables. For example, if tol=0.98, the transformer will remove variables
        that show the same value in at least 98% of the observations.

    missing_values: str, default='raise'
        Whether the missing values should be raised as error, ignored or included as an
        additional value of the variable. Takes values 'raise', 'ignore', 'include'.
        With 'ignore', the proportion of the most frequent value is still calculated
        over all observations, and, if tol=1, variables with a single value besides
        the missing values are dropped. NaN and null are both treated as missing
        values.

    {confirm_variables}

    Attributes
    ----------
    features_to_drop_:
        List with constant and quasi-constant features.

    {variables_}:

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find constant and quasi-constant features.

    {fit_transform}

    {get_support}

    transform:
        Remove constant and quasi-constant features.

    Notes
    -----
    This transformer is a similar concept to the VarianceThreshold from scikit-learn,
    but it evaluates number of unique values instead of variance.

    See Also
    --------
    sklearn.feature_selection.VarianceThreshold

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import DropConstantFeatures
    >>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
    >>>                     x2 = ["a", "a", "b", "c"],
    >>>                     x3 = [True, False, False, True]))
    >>> dcf = DropConstantFeatures()
    >>> dcf.fit_transform(X)
      x2     x3
    0  a   True
    1  a  False
    2  b  False
    3  c   True

    Additionally, you can set the threshold for quasi-constant features:

    >>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
    >>>                      x2 = ["a", "a", "b", "c"],
    >>>                      x3 = [True, False, False, False]))
    >>> dcf = DropConstantFeatures(tol = 0.75)
    >>> dcf.fit_transform(X)
      x2
    0  a
    1  a
    2  b
    3  c

    With polars:

    >>> import polars as pl
    >>> from feature_engine.selection import DropConstantFeatures
    >>> X = pl.DataFrame(dict(x1 = [1,1,1,1],
    >>>                      x2 = ["a", "a", "b", "c"],
    >>>                      x3 = [True, False, False, False]))
    >>> dcf = DropConstantFeatures(tol = 0.75)
    >>> dcf.fit_transform(X)
    shape: (4, 1)
    ┌─────┐
    │ x2  │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ a   │
    │ b   │
    │ c   │
    └─────┘
    """

    def __init__(
        self,
        variables: Variables = None,
        tol: float = 1,
        missing_values: str = "raise",
        confirm_variables: bool = False,
    ):

        if (
            not isinstance(tol, (float, int))
            or isinstance(tol, bool)
            or tol < 0
            or tol > 1
        ):
            raise ValueError(
                f"tol must be a float or integer between 0 and 1. Got {tol} instead."
            )

        if not isinstance(missing_values, str) or missing_values not in [
            "raise",
            "ignore",
            "include",
        ]:
            raise ValueError(
                "missing_values takes only values 'raise', 'ignore' or 'include'. "
                f"Got {missing_values} instead."
            )

        super().__init__(confirm_variables)

        self.tol = tol
        self.variables = _check_variables_input_value(variables)
        self.missing_values = missing_values

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Find constant and quasi-constant features.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The input dataframe. Can be a pandas, polars, or any other dataframe
            supported by narwhals.
        y: None
            y is not needed for this transformer. You can pass y or None.
        """

        # check input dataframe
        nw_X = check_X(X)

        self.variables_ = _select_all_variables(
            X, self.variables, self.confirm_variables
        )

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)

        dropna = self.missing_values != "include"
        n_rows = nw_X.shape[0]

        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            if self.tol == 1:
                self.features_to_drop_ = [
                    feature
                    for feature in self.variables_
                    if X[feature].nunique(dropna=dropna) == 1
                ]
            else:
                # variables with only missing values have no counts; max() is NaN
                # and they are kept.
                self.features_to_drop_ = [
                    feature
                    for feature in self.variables_
                    if X[feature].value_counts(dropna=dropna, sort=False).max() / n_rows
                    >= self.tol
                ]

        else:
            float_vars = [f for f in self.variables_ if nw_X.schema[f].is_float()]

            if self.tol == 1:
                n_unique = nw_X.select(
                    _missing_values(nw.col(f), f in float_vars, dropna).n_unique()
                    for f in self.variables_
                ).row(0)
                drop = [n == 1 for n in n_unique]

            else:
                if nwd.is_polars_dataframe(X) is True:
                    # polars counts the values of all variables in parallel, which is
                    # faster than narwhals.
                    native_ns = nw.get_native_namespace(nw_X)
                    counts = X.select(
                        _missing_values(native_ns.col(f), f in float_vars, dropna)
                        .unique_counts()
                        .max()
                        for f in self.variables_
                    ).row(0)
                else:
                    counts = [
                        _missing_values(nw_X.get_column(f), f in float_vars, dropna)
                        .value_counts(sort=False, name="__count__")
                        .get_column("__count__")
                        .max()
                        for f in self.variables_
                    ]
                # variables with only missing values have no counts (None) and are
                # kept.
                drop = [c is not None and c / n_rows >= self.tol for c in counts]

            self.features_to_drop_ = [
                f for f, d in zip(self.variables_, drop) if d is True
            ]

        # check we are not dropping all the columns in the df
        if len(self.features_to_drop_) == nw_X.shape[1]:
            raise ValueError(
                "The resulting dataframe will have no columns after dropping all "
                "constant or quasi-constant features. Try changing the tol value."
            )

        # save input features
        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_fit2d_1sample"
        ] = "the transformer raises an error when dropping all columns, ok to fail"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def _missing_values(column, is_float: bool, dropna: bool):
    """Treat NaN as missing, like pandas does, and drop the missing values if
    dropna is True. Works with narwhals and polars expressions and series."""
    # polars keeps NaN apart from null.
    if is_float is True:
        column = column.fill_nan(None)
    if dropna is True:
        column = column.drop_nulls()
    return column
