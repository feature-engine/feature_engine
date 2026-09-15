# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Union

import narwhals as nw
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
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
from feature_engine.dataframe_checks import _check_contains_na, check_X_y
from feature_engine.encoding._helper_functions import (
    TARGET_NAME,
    add_target_to_X,
    check_parameter_unseen,
)
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixin,
    CategoricalMethodsMixin,
)
from feature_engine.tags import _return_tags


class WoE:
    def _check_fit_input(self, X: IntoDataFrame, y: IntoSeries):
        """
        Check that X is dataframe, and y a binary series with values 0 and 1.
        """
        nw_X, y = check_X_y(X, y)
        # with pandas, y takes the index of X
        y_nw = add_target_to_X(nw_X, y)[TARGET_NAME]

        # check that y is binary
        if y_nw.n_unique() != 2:
            raise ValueError(
                "This encoder is designed for binary classification. The target "
                "used has more than 2 unique values."
            )

        # if target does not have values 0 and 1, we need to remap, to be able to
        # compute the averages.
        y_min, y_max = y_nw.min(), y_nw.max()
        if y_min != 0 or y_max != 1:
            y_nw = (y_nw != y_min).cast(nw.Int64()).alias("target")

        return X, y_nw.to_native()

    def _calculate_woe(
        self,
        X: IntoDataFrame,
        y: IntoSeries,
        variable: Union[str, int],
    ):
        """
        Return a narwhals dataframe with one row per category of the variable and the
        columns __category__, __pos__ and __neg__, the fraction of positive and
        negative cases, and __woe__, the weight of evidence. Also return whether any
        category has no positive or no negative cases.
        """
        # narwhals expressions need string column names, pandas allows integers
        col = nw.from_native(X, eager_only=True).get_column(variable)
        nw_Xy = add_target_to_X(col.alias("__category__").to_frame(), y)
        total_pos = nw_Xy[TARGET_NAME].sum()
        total_neg = len(nw_Xy) - total_pos

        counts = (
            nw_Xy.group_by("__category__", drop_null_keys=True)
            .agg(nw.col(TARGET_NAME).sum().alias("__pos__"), nw.len().alias("__n__"))
            .sort("__category__")
            .with_columns((nw.col("__n__") - nw.col("__pos__")).alias("__neg__"))
        )
        pos, neg = nw.col("__pos__"), nw.col("__neg__")
        has_zero_counts = bool(counts.select(((pos == 0) | (neg == 0)).any()).item())

        # the WoE is not defined for zero counts, so they are replaced by 0.5
        pos = nw.when(pos == 0).then(0.5).otherwise(pos) / total_pos
        neg = nw.when(neg == 0).then(0.5).otherwise(neg) / total_neg

        woe = counts.select(
            "__category__",
            pos.alias("__pos__"),
            neg.alias("__neg__"),
            (pos / neg).log().alias("__woe__"),
        )
        return woe, has_zero_counts


@Substitution(
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
class WoEEncoder(CategoricalMethodsMixin, CategoricalInitMixin, WoE):
    """
    The WoEEncoder() replaces categories by the weight of evidence
    (WoE). The WoE was used primarily in the financial sector to create credit risk
    scorecards.

    The encoder will encode only categorical variables by default
    (type 'object' or 'categorical'). You can pass a list of variables to encode.
    Alternatively, the encoder will find and encode all categorical variables
    (type 'object' or 'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the weight of evidence for each variable
    (fit). The encoder then transforms the categories into the mapped numbers
    (transform).

    This categorical encoding is exclusive for binary classification.

    **Note**

    The WoE is not defined for categories with no positive or no negative cases. For
    those categories, the encoder replaces the zero count by 0.5, and lists the
    variables in `variables_with_zero_counts_`. Grouping infrequent categories before
    the encoding reduces how often this happens.

    More details in the :ref:`User Guide <woe_encoder>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    {ignore_format}

    {unseen}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the WoE per variable.

    variables_with_zero_counts_:
        List of variables with categories that have no positive or no negative cases.
        For those categories, 0.5 replaces the zero count to calculate the WoE.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the WoE per category, per variable.

    {transform}

    {fit_transform}

    {inverse_transform}

    Notes
    -----
    For details on the calculation of the weight of evidence visit:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    There is a similar implementation in the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    feature_engine.discretisation
    category_encoders.woe.WOEEncoder

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.encoding import WoEEncoder
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4,5], x2 = ["b", "b", "b", "a", "a"]))
    >>> y = pd.Series([0,1,1,1,0])
    >>> woe = WoEEncoder()
    >>> woe.fit(X, y)
    >>> woe.transform(X)
       x1        x2
    0   1  0.287682
    1   2  0.287682
    2   3  0.287682
    3   4 -0.405465
    4   5 -0.405465

    With polars

    >>> import polars as pl
    >>> from feature_engine.encoding import WoEEncoder
    >>> X = pl.DataFrame(dict(x1 = [1,2,3,4,5], x2 = ["b", "b", "b", "a", "a"]))
    >>> y = pl.Series([0,1,1,1,0])
    >>> woe = WoEEncoder()
    >>> woe.fit(X, y)
    >>> woe.transform(X)
    shape: (5, 2)
    ┌─────┬───────────┐
    │ x1  ┆ x2        │
    │ --- ┆ ---       │
    │ i64 ┆ f64       │
    ╞═════╪═══════════╡
    │ 1   ┆ 0.287682  │
    │ 2   ┆ 0.287682  │
    │ 3   ┆ 0.287682  │
    │ 4   ┆ -0.405465 │
    │ 5   ┆ -0.405465 │
    └─────┴───────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        ignore_format: bool = False,
        unseen: str = "ignore",
    ) -> None:

        super().__init__(variables, return_empty, ignore_format)
        check_parameter_unseen(unseen, ["ignore", "raise"])
        self.unseen = unseen

    def fit(self, X: IntoDataFrame, y: IntoSeries):
        """
        Learn the WoE.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: Series.
            Target, must be binary.
        """
        X, y = self._check_fit_input(X, y)
        variables_ = self._check_or_select_variables(X)
        _check_contains_na(X, variables_)

        encoder_dict_ = {}
        variables_with_zero_counts_ = []

        for var in variables_:
            woe, has_zero_counts = self._calculate_woe(X, y, var)
            encoder_dict_[var] = dict(
                zip(woe["__category__"].to_list(), woe["__woe__"].to_list())
            )
            if has_zero_counts is True:
                variables_with_zero_counts_.append(var)

        self.encoder_dict_ = encoder_dict_
        self.variables_with_zero_counts_ = variables_with_zero_counts_
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """Replace categories with the learned parameters.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features].
            The dataframe containing the categories replaced by numbers.
        """

        nw_X = self._check_transform_input_and_state(X)
        _check_contains_na(X, self.variables_)
        X = self._encode(nw_X)
        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        tags_dict["requires_y"] = True
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
