# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause
from collections.abc import Iterable
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.utils.multiclass import check_classification_targets

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
class OOFMeanEncoder(CategoricalMethodsMixin, CategoricalInitMixinNA):
    """
    The OOFMeanEncoder() (Out-of-Fold Mean Encoder) replaces categories with the mean
    target value per category using an out-of-fold cross-validation scheme during
    ``fit_transform()`` to prevent target leakage and overfitting.

    When target encoding is computed on the entire training set and applied directly to
    the same rows, each row's own target value contributes to its encoded value. For
    rare categories or small datasets, this causes severe target leakage where the
    model memorizes training labels rather than generalizing.

    The ``OOFMeanEncoder()`` solves this problem during training by splitting the
    dataset into cross-validation folds (via ``fit_transform()``). For each fold,
    the target encoding mappings are learned strictly using the remaining folds, and
    applied to the held-out fold. Consequently, no row's target value is ever used in
    its own encoding.

    When ``fit()`` is called, or when transforming new unseen/test data with
    ``transform()``, the mappings learned across the entire dataset are used, matching
    the behavior of ``MeanEncoder()``.

    Parameters
    ----------
    {variables}

    cv: int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy for out-of-fold encoding.
        Possible inputs for cv are:

        - None, to use default 5-fold cross-validation
        - int, to specify the number of folds in a KFold or StratifiedKFold
        - CV splitter (e.g., KFold, StratifiedKFold, TimeSeriesSplit)
        - An iterable yielding (train, test) splits as arrays of indices

    smoothing: int, float, str, default=0.0
        Smoothing factor. Should be >= 0. If 0 then no smoothing is applied, and the
        mean target value per category is returned without modification. If 'auto' then
        wi is calculated as described in MeanEncoder and the category is encoded as the
        blended values of the prior and the posterior. If int or float, then wi is
        calculated as ni / (ni + smoothing). Higher values lead to stronger smoothing
        (higher weight of prior).

    {unseen}

    {missing_values}

    {ignore_format}

    {return_empty}

    shuffle: bool, default=False
        Whether to shuffle the data before splitting into folds when ``cv`` is an
        integer.

    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the fold generation when ``shuffle=True`` and
        ``cv`` is an integer. Pass an int for reproducible output.

    stratify: bool, default=False
        Whether to use stratified cross-validation (StratifiedKFold) when ``cv`` is an
        integer. Requires discrete / classification targets.

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the target mean value per category per variable learned on the
        full training set.

    y_prior_:
        Mean of the target across the entire training set.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the target mean value per category, per variable from the entire dataset.

    {fit_transform}

    {inverse_transform}

    {transform}

    See Also
    --------
    feature_engine.encoding.MeanEncoder
    feature_engine.encoding.RareLabelEncoder
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        cv: Union[int, None, Iterable] = 5,
        smoothing: Union[int, float, str] = 0.0,
        unseen: str = "ignore",
        missing_values: str = "raise",
        ignore_format: bool = False,
        return_empty: bool = False,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        stratify: bool = False,
    ) -> None:
        _check_return_empty_is_bool(return_empty)

        super().__init__(variables, missing_values, ignore_format)
        self.return_empty = return_empty

        if (
            not isinstance(smoothing, (str, float, int))
            or (isinstance(smoothing, str) and smoothing != "auto")
            or (isinstance(smoothing, (float, int)) and smoothing < 0)
        ):
            raise ValueError(
                f"smoothing must be greater than 0 or 'auto'. "
                f"Got {smoothing} instead."
            )
        self.smoothing = smoothing

        check_parameter_unseen(unseen, ["ignore", "raise", "encode"])
        self.unseen = unseen

        if cv is not None and isinstance(cv, int) and cv < 2:
            raise ValueError(
                "cv must be an integer greater than 1 or a CV splitter. "
                f"Got {cv} instead."
            )
        self.cv = cv

        if not isinstance(shuffle, bool):
            raise ValueError(
                f"shuffle takes only booleans True and False. Got {shuffle} instead."
            )
        self.shuffle = shuffle
        self.random_state = random_state

        if not isinstance(stratify, bool):
            raise ValueError(
                f"stratify takes only booleans True and False. Got {stratify} instead."
            )
        self.stratify = stratify

    def _calculate_target_means(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        variables: List[Union[str, int]],
        y_prior: float,
    ) -> dict:
        encoder_dict = {}
        if self.smoothing == "auto":
            y_var = y.var(ddof=0)
        for var in variables:
            if self.smoothing == "auto":
                damping = y.groupby(X[var]).var(ddof=0) / y_var
            else:
                damping = self.smoothing
            counts = X[var].value_counts()
            counts.index = counts.index.infer_objects()
            _lambda = counts / (counts + damping)
            encoder_dict[var] = (
                _lambda * y.groupby(X[var], observed=False).mean()
                + (1.0 - _lambda) * y_prior
            ).to_dict()
        return encoder_dict

    def _get_cv_splitter(self, X: pd.DataFrame, y: pd.Series):
        if self.cv is None or isinstance(self.cv, int):
            n_splits = 5 if self.cv is None else self.cv
            if n_splits < 2:
                raise ValueError(
                    f"cv must be an integer greater than 1. Got {self.cv} instead."
                )
            if self.stratify:
                check_classification_targets(y)
                return StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state if self.shuffle else None,
                )
            else:
                return KFold(
                    n_splits=n_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state if self.shuffle else None,
                )
        else:
            return check_cv(self.cv, y=y, classifier=self.stratify)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the mean value of the target for each category of the variable
        across the entire training set.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas series
            The target.
        """
        X, y = check_X_y(X, y)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        y_prior = float(y.mean())
        self.y_prior_ = y_prior

        if self.unseen == "encode":
            self._unseen = y_prior

        self.encoder_dict_ = self._calculate_target_means(
            X, y, variables_, y_prior
        )

        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit to data, then transform it with out-of-fold target encoding.

        Fits the transformer using the full training set (populating `encoder_dict_`
        and attributes for subsequent `transform` calls), then transforms `X` using
        out-of-fold target encoding where each row is encoded using only the
        other folds.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas series
            The target.

        Returns
        -------
        X_tr: pandas dataframe of shape = [n_samples, n_features]
            The out-of-fold encoded dataframe.
        """
        X, y = check_X_y(X, y)

        self.fit(X, y)

        if not self.variables_:
            return X

        splitter = self._get_cv_splitter(X, y)

        X_tr = X.copy()
        n_samples = len(X)

        encoded_cols = {
            var: np.full(n_samples, np.nan, dtype=float) for var in self.variables_
        }

        for train_idx, val_idx in splitter.split(X, y):
            X_train_f = X.iloc[train_idx]
            y_train_f = y.iloc[train_idx]
            X_val_f = X.iloc[val_idx]

            y_prior_f = float(y_train_f.mean())

            fold_mappings = self._calculate_target_means(
                X_train_f, y_train_f, self.variables_, y_prior_f
            )

            for var in self.variables_:
                mapped_val = X_val_f[var].map(fold_mappings[var])
                if self.unseen == "encode":
                    mapped_val = mapped_val.fillna(y_prior_f)
                encoded_cols[var][val_idx] = mapped_val.to_numpy(dtype=float)

        for var in self.variables_:
            X_tr[var] = encoded_cols[var]

        if self.unseen != "encode":
            self._check_nan_values_after_transformation(X_tr)

        return X_tr

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
