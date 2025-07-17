# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause
from typing import List, Union

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
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

    Check also the related transformers in the the open-source package
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
        missing_values: str = "raise",
        ignore_format: bool = False,
        unseen: str = "ignore",
        smoothing: Union[int, float, str] = 0.0,
    ) -> None:
        super().__init__(variables, missing_values, ignore_format)
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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the mean value of the target for each category of the variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to be encoded.

        y: pandas series
            The target.
        """

        X, y = check_X_y(X, y)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        y_prior = y.mean()

        if self.unseen == "encode":
            self._unseen = y_prior

        if self.smoothing == "auto":
            y_var = y.var(ddof=0)
        for var in variables_:
            if self.smoothing == "auto":
                damping = y.groupby(X[var]).var(ddof=0) / y_var
            else:
                damping = self.smoothing
            counts = X[var].value_counts()
            counts.index = counts.index.infer_objects()
            _lambda = counts / (counts + damping)
            self.encoder_dict_[var] = (
                _lambda * y.groupby(X[var], observed=False).mean()
                + (1.0 - _lambda) * y_prior
            ).to_dict()

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert the encoded variable back to the original values.

        Note that if unseen was set to 'encode', then this method is not implemented.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: pandas dataframe of shape = [n_samples, n_features].
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
