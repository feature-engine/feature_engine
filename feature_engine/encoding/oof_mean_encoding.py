# Authors: hx-max
# License: BSD 3 clause
"""Out-of-fold mean (target) encoding for categorical variables."""

import warnings
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

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

_MAX_CLASSIFICATION_LEVELS = 20


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
class OOFMeanEncoder(CategoricalMethodsMixin, CategoricalInitMixinNA):
    """
    The OOFMeanEncoder() replaces categories with the mean of the target, computed
    **out-of-fold**, which prevents target leakage in the training data.

    Mean (target) encoding — replacing a category by the mean target value observed
    for that category — is a powerful way of encoding high cardinality categorical
    variables, but it has a well known flaw: when the encoding is estimated and
    applied on the same rows, every row contributes its own target value to its own
    encoding. For high cardinality variables (many categories with few observations)
    the encoding becomes a copy of the target, and models trained on it overfit
    spectacularly: they learn that the encoded feature predicts the target, while on
    new data the same feature is far less informative.

    This transformer fixes that by computing the mapping **out-of-fold** during
    `fit_transform`: the data is split into `cv` folds; for every fold, the category
    mean is estimated **only on the remaining folds** and then used to encode the
    held-out rows. As a result, no row ever uses its own target value to build its
    own encoding, while the *statistics* the encoding relies on are still estimated
    from the whole data set.

    `fit` still learns the mapping from the entire data set. That mapping is used to
    encode new data during `transform` (and, for consistency, by `inverse_transform`).
    Therefore the standard usage is::

        X_train_encoded = encoder.fit_transform(X_train, y_train)  # leak free
        X_test_encoded = encoder.transform(X_test)                 # full mapping

    Smoothing (shrinkage of rare category means towards the prior) works exactly as
    in :class:`MeanEncoder`: with `smoothing='auto'` the weight given to the category
    mean is estimated from the ratio of the target variance within the category to
    the target variance of the data set, and with `smoothing=int or float` the weight
    is `n_i / (n_i + smoothing)`.

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode. Alternatively, the
    encoder will find and encode all categorical variables (type 'object' or
    'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as
    well. The procedure is identical, you can either enter the list of variables to
    encode, or the transformer will automatically select all variables.

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

    cv: int, default=5
        Number of folds used to generate the out-of-fold encodings. Must be >= 2.
        Larger values mean less bias in the training encodings but more computation.

    shuffle: bool, default=False
        Whether to shuffle the data before splitting it into folds. If `True`,
        `random_state` can be used for reproducibility.

    random_state: int, default=None
        Seed used when `shuffle=True`.

    stratify: bool, default=None
        Whether to use stratified folds, that is, folds that preserve the proportion
        of target classes. If `None` (default) the encoder decides automatically:
        stratified folds are used when the target looks categorical (boolean, integer,
        or float with integral values, with at most 20 distinct values), and plain
        K-Fold folds otherwise. Stratification is skipped, and plain K-Fold folds are
        used instead, when any class has fewer observations than folds.

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the target mean value per category per variable, learnt on the
        entire data set. Used by `transform`.

    fold_mappings_:
        List with the mapping (dictionary of category to target mean value, per
        variable) learnt in each fold during `fit_transform`.

    fold_priors_:
        List with the target mean of the training part of each fold.

    y_prior_:
        Target mean of the entire data set.

    n_splits_:
        Number of folds actually used in `fit_transform`.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the target mean value per category, per variable, on the entire data
        set.

    {fit_transform}

    {transform}

    {inverse_transform}

    Notes
    -----
    `fit_transform` is the method that produces leak free encodings for the training
    data, because `transform` can only use the mapping learnt on the whole data set
    (which includes the target of every row). In other words, encodings produced by
    `fit_transform` and `transform` differ slightly, which is expected and is the
    whole point of the out-of-fold scheme.

    Categories that appear in a validation fold but not in the remaining folds cannot
    be mapped, because their mean is unknown. During `fit_transform` those rows are
    encoded with the target mean of the remaining folds (the fold prior), so that the
    output never contains missing values. NAN can still be introduced by `transform`
    for categories that were not present in the training data, exactly as in
    :class:`MeanEncoder`.

    Check also the related transformers in the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    feature_engine.encoding.MeanEncoder
    feature_engine.encoding.RareLabelEncoder
    category_encoders.target_encoder.TargetEncoder
    category_encoders.m_estimate.MEstimateEncoder

    References
    ----------
    .. [1] Micci-Barreca D. "A Preprocessing Scheme for High-Cardinality Categorical
       Attributes in Classification and Prediction Problems". ACM SIGKDD Explorations
       Newsletter, 2001. https://dl.acm.org/citation.cfm?id=507538
    .. [2] Prokhorenkova L. et al. "CatBoost: unbiased boosting with categorical
       features". NeurIPS, 2018. https://arxiv.org/abs/1706.09516

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.encoding import OOFMeanEncoder
    >>> X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": ["c", "c", "c", "b", "a"]})
    >>> y = pd.Series([0, 1, 1, 1, 0])
    >>> oof = OOFMeanEncoder(cv=2)
    >>> Xt = oof.fit_transform(X, y)
    >>> oof.transform(X)
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
        cv: int = 5,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        stratify: Union[bool, None] = None,
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

        if not isinstance(cv, int) or isinstance(cv, bool) or cv < 2:
            raise ValueError(f"cv must be an integer greater than 1. Got {cv} instead.")
        self.cv = cv

        if not isinstance(shuffle, bool):
            raise ValueError(
                f"shuffle takes only booleans True and False. Got {shuffle} instead."
            )
        self.shuffle = shuffle

        if stratify is not None and not isinstance(stratify, bool):
            raise ValueError(
                f"stratify takes only booleans True and False, or None. "
                f"Got {stratify} instead."
            )
        self.stratify = stratify

        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the mean value of the target for each category of the variable, using
        the entire data set. This mapping is the one used by `transform`.

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

        self.y_prior_ = float(y.mean())

        if self.unseen == "encode":
            self._unseen = self.y_prior_

        self.encoder_dict_ = self._learn_mapping(
            X, y, variables_, prior=self.y_prior_
        )

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Learn the mean value of the target per category **out-of-fold** and encode the
        training data with it, so that no observation is encoded with a statistic that
        contains its own target value.

        This method also leaves the transformer fitted with the mapping learnt on the
        entire data set, so `transform` can be used afterwards on new data.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to be encoded.

        y: pandas series
            The target.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the categories replaced by the out-of-fold target mean
            values.
        """

        # learn the mapping on the whole data set: it is used by transform.
        self.fit(X, y)

        X_oof = X.copy()

        # Out-of-fold values are floats. Re-initialise the columns to encode as
        # float64 so that (a) assigning floats never fails, whatever the original
        # dtype is (object, category, integer), and (b) the output is numeric.
        # Every row is assigned in exactly one fold, so no value is left as NA.
        for var in self.variables_:
            X_oof[var] = pd.Series(np.nan, index=X_oof.index, dtype="float64")

        splits = self._make_splits(X, y)
        self.n_splits_ = len(splits)
        self.fold_mappings_ = []
        self.fold_priors_ = []

        for train_index, valid_index in splits:
            X_fold, y_fold = X.iloc[train_index], y.iloc[train_index]
            fold_prior = float(y_fold.mean())

            fold_mapping = self._learn_mapping(
                X_fold, y_fold, self.variables_, prior=fold_prior
            )

            for var in self.variables_:
                col = X_oof.columns.get_loc(var)
                values = X.iloc[valid_index][var].map(fold_mapping[var])

                if values.isnull().any():
                    if self.unseen == "raise":
                        raise ValueError(
                            "During the out-of-fold encoding, categories that were not "
                            f"present in the training folds were found in variable "
                            f"'{var}'. Set unseen to 'ignore' or 'encode' to encode "
                            "them with the prior of the training folds instead."
                        )
                    values = values.fillna(fold_prior)

                X_oof.iloc[valid_index, col] = values.to_numpy(dtype="float64")

            self.fold_mappings_.append(fold_mapping)
            self.fold_priors_.append(fold_prior)

        return X_oof

    def _learn_mapping(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        variables_: List[Union[str, int]],
        prior: float,
    ) -> dict:
        """Return the mapping of category to target mean value, per variable."""

        mapping = {}

        if self.smoothing == "auto":
            y_var = y.var(ddof=0)

        for var in variables_:
            if self.smoothing == "auto":
                damping = (
                    y.groupby(X[var], observed=False).var(ddof=0) / y_var
                    if y_var > 0
                    else 0
                )
            else:
                damping = self.smoothing

            counts = X[var].value_counts()
            counts.index = counts.index.infer_objects()
            _lambda = counts / (counts + damping)

            mapping[var] = (
                _lambda * y.groupby(X[var], observed=False).mean()
                + (1.0 - _lambda) * prior
            ).to_dict()

        return mapping

    def _make_splits(self, X: pd.DataFrame, y: pd.Series) -> List[tuple]:
        """Create the list of (train, validation) positional indices."""

        stratified = self._resolve_stratify(y)

        if stratified:
            classes = pd.Series(np.asarray(y)).value_counts()
            smallest_class = int(classes.min()) if not classes.empty else 0
            if smallest_class < self.cv:
                warnings.warn(
                    "The smallest target class has fewer observations than folds "
                    f"({smallest_class} < {self.cv}), therefore stratified folds are "
                    "not possible and plain K-Fold folds are used instead."
                )
                stratified = False

        splitter_cls = StratifiedKFold if stratified else KFold

        kwargs: dict = {"n_splits": self.cv}
        if self.shuffle:
            kwargs["shuffle"] = True
            kwargs["random_state"] = self.random_state

        splitter = splitter_cls(**kwargs)

        if stratified:
            return list(splitter.split(X, y))
        return list(splitter.split(X))

    def _resolve_stratify(self, y: pd.Series) -> bool:
        """Return True if the target looks categorical and folds can be stratified."""

        if self.stratify is not None:
            return self.stratify

        y_array = np.asarray(y)

        if pd.Series(y_array).nunique(dropna=True) > _MAX_CLASSIFICATION_LEVELS:
            return False

        if y_array.dtype.kind in "biu":
            return True

        if y_array.dtype.kind == "f":
            return bool(np.allclose(y_array, np.round(y_array)))

        return False

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
