from collections import defaultdict
from typing import List, Tuple, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame

from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_missing_values,
)
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
    _missing_values_docstring,
    _variables_all_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_contains_na, check_X
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags

from .base_selection_functions import _select_all_variables

Variables = Union[None, int, str, List[Union[str, int]]]

# from 2**53 on, float64 can't tell consecutive integers apart.
_MAX_EXACT_INTEGER = 2**53


@Substitution(
    confirm_variables=_confirm_variables_docstring,
    variables=_variables_all_docstring,
    missing_values=_missing_values_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class DropDuplicateFeatures(BaseSelector):
    """
    DropDuplicateFeatures() finds and removes duplicated features in a dataframe.

    Duplicated features are identical features, regardless of the variable or column
    name. If they show the same values for every observation, then they are considered
    duplicated. Numbers are compared by value, so an integer and a float variable, or
    a boolean and a 0/1 variable, can be duplicated. String and categorical variables
    are compared by their values. Missing values are equal to each other.

    This transformer works with numerical and categorical variables. The user can
    indicate a list of variables to examine. Alternatively, the transformer will
    evaluate all the variables in the dataset.

    The transformer will first identify and store the duplicated variables. Next, the
    transformer will drop these variables from a dataframe.

    More details in the :ref:`User Guide <drop_duplicate>`.

    Parameters
    ----------
    {variables}

    {missing_values}

    {confirm_variables}

    Attributes
    ----------
    features_to_drop_:
        Set with the duplicated features that will be dropped.

    duplicated_feature_sets_:
        Groups of duplicated features. Each list is a group of duplicated features.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find duplicated features.

    {fit_transform}

    {get_support}

    transform:
        Remove duplicated features.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import DropDuplicateFeatures
    >>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
    >>>                     x2 = [1,1,1,1],
    >>>                     x3 = [True, False, False, False]))
    >>> ddf = DropDuplicateFeatures()
    >>> ddf.fit_transform(X)
        x1     x3
    0   1   True
    1   1  False
    2   1  False
    3   1  False

    With polars:

    >>> import polars as pl
    >>> from feature_engine.selection import DropDuplicateFeatures
    >>> X = pl.DataFrame(dict(x1 = [1,1,1,1],
    >>>                     x2 = [1,1,1,1],
    >>>                     x3 = [True, False, False, False]))
    >>> ddf = DropDuplicateFeatures()
    >>> ddf.fit_transform(X)
    shape: (4, 2)
    ┌─────┬───────┐
    │ x1  ┆ x3    │
    │ --- ┆ ---   │
    │ i64 ┆ bool  │
    ╞═════╪═══════╡
    │ 1   ┆ true  │
    │ 1   ┆ false │
    │ 1   ┆ false │
    │ 1   ┆ false │
    └─────┴───────┘
    """

    def __init__(
        self,
        variables: Variables = None,
        missing_values: str = "ignore",
        confirm_variables: bool = False,
    ):
        _check_param_missing_values(missing_values)

        super().__init__(confirm_variables)

        self.variables = _check_variables_input_value(variables)
        self.missing_values = missing_values

    def fit(self, X: IntoDataFrame, y=None):
        """
        Find duplicated features.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The input dataframe.
        y: None
            y is not needed for this transformer. You can pass y or None.
        """
        nw_X = check_X(X)

        self.variables_ = _select_all_variables(
            X, self.variables, self.confirm_variables
        )

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)

        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            fingerprints = _pandas_fingerprints(X, self.variables_, nw_X.schema)
        else:
            fingerprints = _polars_fingerprints(nw_X, self.variables_)

        # features with the same fingerprint are duplicates; the first one is kept.
        features_hashmap = defaultdict(list)
        for feature, fingerprint in zip(self.variables_, fingerprints):
            features_hashmap[fingerprint].append(feature)

        duplicates = [group for group in features_hashmap.values() if len(group) > 1]
        self.duplicated_feature_sets_ = [set(group) for group in duplicates]
        self.features_to_drop_ = {
            feature for group in duplicates for feature in group[1:]
        }

        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def _value_kind(dtype) -> Union[str, Tuple[str, bool]]:
    """Return the kind of values in a column. Only columns of the same kind can be
    duplicates: numbers (booleans included), datetimes, durations and other values.
    """
    if dtype.is_numeric() or dtype == nw.Boolean:
        return "number"
    if dtype == nw.Datetime:
        # instants are compared in UTC, so the time zone only separates naive from
        # time zone aware datetimes.
        return ("datetime", dtype.time_zone is None)
    if dtype == nw.Duration:
        return "duration"
    return "other"


def _pandas_fingerprints(
    X: IntoDataFrame, variables: List[Union[str, int]], schema: nw.Schema
) -> List:
    """Return a key per variable that is equal for variables with the same values."""
    hash_pandas_object = nw.get_native_namespace(X).util.hash_pandas_object
    fingerprints: List = []
    for variable in variables:
        kind = _value_kind(schema[variable])
        if kind == "number":
            values = X[variable].to_numpy(dtype="float64", na_value=np.nan)
            is_null = np.isnan(values)
            max_value = np.abs(values).max(initial=0, where=~is_null)
            if schema[variable].is_integer() and max_value >= _MAX_EXACT_INTEGER:
                kind = "large integer"
                values = hash_pandas_object(X[variable], index=False).to_numpy()
            else:
                # adding 0.0 turns -0.0 into 0.0, and NaNs may differ in their bits.
                values = values + 0.0
                values[is_null] = np.nan
        elif kind == "other":
            is_null = X[variable].isna().to_numpy()
            values = hash_pandas_object(X[variable], index=False).to_numpy()
        else:
            unit = "timedelta64[ns]" if kind == "duration" else "datetime64[ns]"
            values = X[variable].to_numpy(dtype=unit)
            is_null = np.isnat(values)

        # missing values are equal whatever the data type of the column.
        if is_null.all():
            fingerprints.append("null")
        else:
            fingerprints.append((kind, hash(values.tobytes())))
    return fingerprints


def _polars_fingerprints(nw_X: nw.DataFrame, variables: List[Union[str, int]]) -> List:
    """Return a key per variable that is equal for variables with the same values."""
    # narwhals has no hash function, so the values are hashed by polars.
    pl = nw.get_native_namespace(nw_X)
    schema = nw_X.schema
    kinds = [_value_kind(schema[variable]) for variable in variables]

    integers = [variable for variable in variables if schema[variable].is_integer()]
    large_integers = []
    if len(integers) > 0:
        max_values = nw_X.select(
            nw.col(*integers).cast(nw.Float64).abs().max().fill_null(0)
        ).row(0)
        large_integers = [
            variable
            for variable, max_value in zip(integers, max_values)
            if max_value >= _MAX_EXACT_INTEGER
        ]

    columns = []
    for i, variable in enumerate(variables):
        column = pl.col(variable)
        if variable in large_integers:
            kinds[i] = "large integer"
        elif kinds[i] == "number":
            column = column.cast(pl.Float64).fill_nan(None)
        elif kinds[i] == "other":
            if schema[variable] in (nw.Categorical, nw.Enum):
                column = column.cast(pl.String)
        else:
            column = column.dt.cast_time_unit("ns")
        columns.append(column)

    # odd weights make the sum of the row hashes depend on the order of the rows.
    weights = np.random.default_rng(0).integers(
        0, 2**63, size=nw_X.shape[0], dtype=np.uint64
    )
    weights = pl.lit(pl.Series(weights * np.uint64(2) + np.uint64(1)))
    row = (
        nw_X.to_native()
        .select(
            *[
                (column.hash() * weights).sum().alias(f"__hash_{i}__")
                for i, column in enumerate(columns)
            ],
            *[
                column.null_count().alias(f"__null_count_{i}__")
                for i, column in enumerate(columns)
            ],
        )
        .row(0)
    )
    n_variables = len(variables)
    hashes, null_counts = row[:n_variables], row[n_variables:]

    return [
        "null" if null_count == nw_X.shape[0] else (kind, row_hash)
        for kind, row_hash, null_count in zip(kinds, hashes, null_counts)
    ]
