# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause
from typing import List, Union

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
from feature_engine.dataframe_checks import check_X_y
from feature_engine.encoding._helper_functions import check_parameter_unseen
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixinNA,
    CategoricalMethodsMixin,
)

_unseen_docstring = (
    _unseen_docstring
    + """ If `'encode'`, unseen categories will be encoded with the prior."""
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
class MeanEncoder(CategoricalMethodsMixin, CategoricalInitMixinNA):
    """
    The MeanEncoder() replaces categories by the mean value of the target for each
    category.

    For example in the variable colour, if the mean of the target for blue, red
    and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 0.5, red by 0.8
    and grey by 0.1.

    For rare categories, i.e., those with few observations, the mean target value
    might be less reliable. To mitigate poor estimates returned for rare categories,
    the mean target value can be determined as a mixture of the target mean value for
    the entire data set (also called the prior) and the mean target value for the
    category (the posterior), weighted by the number of observations:

    .. math::

        mapping = (w_i) posterior + (1-w_i) prior

    where the weight is calculated as:

      .. math::

        w_i = n_i t / (s + n_i t)

    In the previous equation, t is the target variance in the entire dataset, s is the
    target variance within the category and n is the number of observations for the
    category.

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode. Alternatively, the
    encoder will find and encode all categorical variables (type 'object' or
    'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the numbers for each variable (fit). The
    encoder then replaces the categories with those numbers (transform).

    More details in the :ref:`User Guide <mean_encoder>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    {missing_values}

    {ignore_format}

    {unseen}

    smoothing: int, float, str, default=0.0
        Smoothing factor. Should be >= 0. If 0 then no smoothing is applied, and the
        mean target value per category is returned without modification. If 'auto' then
        wi is calculated as described above and the category is encoded as the blended
        values of the prior and the posterior. If int or float, then the wi is
        calculated as ni / (ni+smoothing). Higher values lead to stronger smoothing
        (higher weight of prior).

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the target mean value per category per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the target mean value per category, per variable.

    {fit_transform}

    {inverse_transform}

    {transform}

    Notes
    -----
    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    Check also the related transformers in the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    category_encoders.target_encoder.TargetEncoder
    category_encoders.m_estimate.MEstimateEncoder

    References
    ----------
    .. [1] Micci-Barreca D. "A Preprocessing Scheme for High-Cardinality Categorical
       Attributes in Classification and Prediction Problems". ACM SIGKDD Explorations
       Newsletter, 2001. https://dl.acm.org/citation.cfm?id=507538

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.encoding import MeanEncoder
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4,5], x2 = ["c", "c", "c", "b", "a"]))
    >>> y = pd.Series([0,1,1,1,0])
    >>> me = MeanEncoder()
    >>> me.fit(X,y)
    >>> me.transform(X)
       x1        x2
    0   1  0.666667
    1   2  0.666667
    2   3  0.666667
    3   4  1.000000
    4   5  0.000000
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        missing_values: str = "raise",
        ignore_format: bool = False,
        unseen: str = "ignore",
        smoothing: Union[int, float, str] = 0.0,
    ) -> None:
        _check_return_empty_is_bool(return_empty)

        super().__init__(variables, missing_values, ignore_format)
        self.return_empty = return_empty
        if (
            not isinstance(smoothing, (str, float, int))
            or isinstance(smoothing, str)
            and (smoothing != "auto")
        ) or (isinstance(smoothing, (float, int)) and smoothing < 0):
            raise ValueError(
                f"smoothing must be greater than 0 or 'auto'. "
                f"Got {smoothing} instead."
            )
        self.smoothing = smoothing
        check_parameter_unseen(unseen, ["ignore", "raise", "encode"])
        self.unseen = unseen

    def fit(self, X: IntoDataFrame, y: IntoSeries):
        """
        Learn the mean value of the target for each category of the variable.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to be encoded.

        y: Series
            The target.
        """

        X, y = check_X_y(X, y)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        # benchmarked at 10k-100k rows x 1-10 cols x 5-50 categories: a pure
        # narwhals fit() ran ~1.5x-2.9x slower than pandas-native here, worse
        # at low column counts (the common case), so pandas keeps its native
        # groupby/value_counts fast path and only polars (and other
        # backends) go through narwhals.
        is_pandas = nwd.is_pandas_dataframe(X)

        if is_pandas is True:
            y_prior = y.mean()

            if self.unseen == "encode":
                self._unseen = y_prior

            if self.smoothing == "auto":
                y_var = y.var(ddof=0)

            for var in variables_:
                # y may be a Series (aligned with X by index, per check_X_y)
                # or a numpy array (list/array-like y goes through sklearn's
                # column_or_1d, which has no .groupby()) - pair the latter
                # with X[var] positionally via assign() instead.
                if nwd.is_pandas_series(y):
                    target, group_keys = y, X[var]
                else:
                    target_name = "__feature_engine_mean_target__"
                    paired = X[[var]].assign(**{target_name: y})
                    target, group_keys = paired[target_name], paired[var]

                if self.smoothing == "auto":
                    damping = target.groupby(group_keys).var(ddof=0) / y_var
                else:
                    damping = self.smoothing
                counts = X[var].value_counts()
                counts.index = counts.index.infer_objects()
                _lambda = counts / (counts + damping)
                self.encoder_dict_[var] = (
                    _lambda * target.groupby(group_keys, observed=False).mean()
                    + (1.0 - _lambda) * y_prior
                ).to_dict()
        else:
            nw_X = nw.from_native(X, eager_only=True)
            target_name = "__feature_engine_mean_target__"
            if nwd.is_into_series(y):
                y_nw = nw.from_native(y, series_only=True).alias(target_name)
            else:
                y_nw = nw.new_series(
                    name=target_name, values=y, backend=nw_X.implementation
                )
            nw_Xy = nw_X.with_columns(y_nw)

            y_prior = y_nw.mean()

            if self.unseen == "encode":
                self._unseen = y_prior

            if self.smoothing == "auto":
                y_var = y_nw.var(ddof=0)

            for var in variables_:
                aggs = [
                    nw.col(target_name).mean().alias("__mean__"),
                    nw.col(target_name).len().alias("__count__"),
                ]
                if self.smoothing == "auto":
                    aggs.append(nw.col(target_name).var(ddof=0).alias("__var__"))
                grouped = nw_Xy.group_by(var, drop_null_keys=True).agg(*aggs)
                stats = grouped.to_dict(as_series=False)

                mapping = {}
                for i, cat in enumerate(stats[var]):
                    n_i = stats["__count__"][i]
                    if self.smoothing == "auto":
                        damping = stats["__var__"][i] / y_var
                    else:
                        damping = self.smoothing
                    _lambda = n_i / (n_i + damping)
                    mapping[cat] = (
                        _lambda * stats["__mean__"][i] + (1.0 - _lambda) * y_prior
                    )
                self.encoder_dict_[var] = mapping

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def inverse_transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """Convert the encoded variable back to the original values.

        Note that if unseen was set to 'encode', then this method is not implemented.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: dataframe of shape = [n_samples, n_features].
            The un-transformed dataframe, with the categorical variables containing the
            original values.
        """

        if self.unseen == "encode":
            raise NotImplementedError(
                "inverse_transform is not implemented for this transformer when "
                "`unseen='encode'`."
            )
        else:
            return super().inverse_transform(X)

    def _more_tags(self):
        tags_dict = super()._more_tags()
        tags_dict["requires_y"] = True
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
