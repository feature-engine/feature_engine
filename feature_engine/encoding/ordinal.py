# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _missing_values_docstring,
    _return_empty_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import (
    _ignore_format_docstring,
    _unseen_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
    _transform_encoders_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X, check_X_y
from feature_engine.encoding._helper_functions import check_parameter_unseen
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixinNA,
    CategoricalMethodsMixin,
)

_unseen_docstring = (
    _unseen_docstring + """ If `'encode'`, unseen categories will be encoded as -1."""
)


@Substitution(
    missing_values=_missing_values_docstring,
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
    return_empty=_return_empty_docstring,
    unseen=_unseen_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    transform=_transform_encoders_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class OrdinalEncoder(CategoricalMethodsMixin, CategoricalInitMixinNA):
    """
    The OrdinalEncoder() replaces categories by ordinal numbers
    (0, 1, 2, 3, etc). The numbers can be ordered based on the mean of the target
    per category, or assigned arbitrarily.

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode. Alternatively, the
    encoder will find and encode all categorical variables (type 'object' or
    'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the numbers for each variable (fit). The
    encoder then transforms the categories to the mapped numbers (transform).

    More details in the :ref:`User Guide <ordinal_encoder>`.

    Parameters
    ----------
    encoding_method: str, default='ordered'
        Desired method of encoding.

        **'ordered'**: the categories are numbered in ascending order according to
        the target mean value per category.

        **'arbitrary'**: categories are numbered arbitrarily.

    {variables}

    {return_empty}

    {missing_values}

    {ignore_format}


    {unseen}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the ordinal number per category, per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find the integer to replace each category in each variable.

    {fit_transform}

    {inverse_transform}

    {transform}

    Notes
    -----
    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    There is a similar implementation in the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    category_encoders.ordinal.OrdinalEncoder

    References
    ----------
    Encoding into integers ordered following target mean was discussed in the following
    talk at PyData London 2017:

    .. [1] Galli S. "Machine Learning in Financial Risk Assessment".
        https://www.youtube.com/watch?v=KHGGlozsRtA

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.encoding import OrdinalEncoder
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4], x2 = ["c", "a", "b", "c"]))
    >>> y = pd.Series([0,1,1,0])
    >>> od = OrdinalEncoder(encoding_method='arbitrary')
    >>> od.fit(X)
    >>> od.transform(X)
       x1  x2
    0   1   0
    1   2   1
    2   3   2
    3   4   0

    You can also consider the order of the target variable:

    >>> y = pd.Series([1,0,1,1])
    >>> od = OrdinalEncoder(encoding_method='ordered')
    >>> od.fit(X, y)
    >>> od.transform(X)
       x1  x2
    0   1   2
    1   2   0
    2   3   1
    3   4   2
    """

    def __init__(
        self,
        encoding_method: str = "ordered",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        missing_values: str = "raise",
        ignore_format: bool = False,
        unseen: str = "ignore",
    ) -> None:

        if encoding_method not in ["ordered", "arbitrary"]:
            raise ValueError(
                "encoding_method takes only values 'ordered' and 'arbitrary'"
            )

        check_parameter_unseen(unseen, ["ignore", "raise", "encode"])
        _check_return_empty_is_bool(return_empty)

        super().__init__(variables, missing_values, ignore_format)
        self.encoding_method = encoding_method
        self.unseen = unseen
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """Learn the numbers to be used to replace the categories in each
        variable.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to be encoded.

        y: Series, default=None
            The Target. Can be None if `encoding_method='arbitrary'`.
            Otherwise, y needs to be passed when fitting the transformer.
        """

        if self.encoding_method == "ordered":
            X, y = check_X_y(X, y)
        else:
            X = check_X(X)

        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        # benchmarked at 10k-100k rows x 1-10 cols x 5-50 categories: a pure
        # narwhals fit() ran 5x-18x slower than pandas-native here (unlike
        # the encode/transform hot path in base_encoder.py, which is only
        # ~1.1x), so pandas keeps its native groupby/unique fast path and
        # only polars (and other backends) go through narwhals.
        is_pandas = nwd.is_pandas_dataframe(X)

        if is_pandas is True:
            for var in variables_:
                if self.encoding_method == "ordered":
                    if nwd.is_pandas_series(y):
                        t = y.groupby(X[var], observed=False).mean()  # type: ignore
                    else:
                        # y is a numpy array here (e.g. list/array-like input
                        # went through sklearn's column_or_1d instead of
                        # check_X_y's Series passthrough); it has no
                        # .groupby(), so pair it with X[var] positionally via
                        # assign() instead - this also matches how the
                        # narwhals branch below handles a non-Series y.
                        t = (
                            X[[var]]
                            .assign(__feature_engine_ordinal_target__=y)
                            .groupby(var, observed=False)[
                                "__feature_engine_ordinal_target__"
                            ]
                            .mean()
                        )
                    t = t.sort_values(ascending=True).index
                elif self.encoding_method == "arbitrary":
                    if self.missing_values == "ignore":
                        t = X[var].dropna().unique()
                    else:
                        t = X[var].unique()
                else:
                    raise ValueError(
                        "Unrecognized value for encoding_method. It should be "
                        f"'arbitrary' or 'frequency'. Got {self.encoding_method} "
                        "instead."
                    )
                self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}
        else:
            nw_X = nw.from_native(X, eager_only=True)

            if self.encoding_method == "ordered":
                # y may already be a Series (polars, from check_X_y) or a
                # plain numpy array (sklearn's column_or_1d path for
                # list/array input) - normalise both to a narwhals Series
                # aliased to a sentinel name, then attach it to the full
                # frame once so every variable's group_by below can reuse it.
                target_name = "__feature_engine_ordinal_target__"
                if nwd.is_into_series(y):
                    y_nw = nw.from_native(y, series_only=True).alias(target_name)
                else:
                    y_nw = nw.new_series(
                        name=target_name, values=y, backend=nw_X.implementation
                    )
                nw_Xy = nw_X.with_columns(y_nw)

            for var in variables_:
                if self.encoding_method == "ordered":
                    # sort by (mean, category): group_by's own order isn't
                    # guaranteed across backends, and this tie-break on the
                    # category itself reproduces pandas' groupby(sort=True)
                    # + stable sort_values behavior for categories with equal
                    # target means.
                    t = (
                        nw_Xy.group_by(var, drop_null_keys=True)
                        .agg(nw.col(target_name).mean())
                        .sort([target_name, var])
                        .get_column(var)
                        .to_list()
                    )
                elif self.encoding_method == "arbitrary":
                    col = nw_X.get_column(var)
                    if self.missing_values == "ignore":
                        col = col.drop_nulls()
                    t = col.unique(maintain_order=True).to_list()
                else:
                    raise ValueError(
                        "Unrecognized value for encoding_method. It should be "
                        f"'arbitrary' or 'frequency'. Got {self.encoding_method} "
                        "instead."
                    )
                self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        if self.unseen == "encode":
            self._unseen = -1

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self
