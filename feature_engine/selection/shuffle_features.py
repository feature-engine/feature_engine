from types import GeneratorType
from typing import List, MutableSequence, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _check_sample_weight, check_random_state

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
    _cv_docstring,
    _estimator_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _initial_model_performance_docstring,
    _scoring_docstring,
    _threshold_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X_y
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags

from .base_selection_functions import _select_numerical_variables

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    variables=_variables_numerical_docstring,
    confirm_variables=_confirm_variables_docstring,
    initial_model_performance_=_initial_model_performance_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class SelectByShuffling(BaseSelector):
    """
    SelectByShuffling() selects features by determining the drop in machine learning
    model performance when each feature's values are randomly shuffled.

    If the variables are important, a random permutation of their values will
    decrease dramatically the machine learning model performance. Contrarily, the
    permutation of the values should have little to no effect on the model performance
    metric we are assessing if the feature is not predictive.

    The SelectByShuffling() first trains a machine learning model utilising all
    features. Next, it shuffles the values of 1 feature, obtains a prediction with the
    pre-trained model, and determines the performance drop (if any). If the drop in
    performance is bigger than a threshold then the feature is retained, otherwise
    removed. It continues until all features have been shuffled and examined.

    The user can determine the model for which performance drop after feature shuffling
    should be assessed. The user also determines the threshold in performance under
    which a feature will be removed, and the performance metric to evaluate.

    Model training and performance calculation are done with cross-validation.

    More details in the :ref:`User Guide <feature_shuffling>`.

    Parameters
    ----------
    {estimator}

    {variables}

    {scoring}

    {threshold}

    {cv}

    random_state: int, default=None
        Controls the randomness when shuffling features.

    {confirm_variables}

    Attributes
    ----------
    {initial_model_performance_}

    performance_drifts_:
        Dictionary with the performance drift per shuffled feature.

    performance_drifts_std_:
        Dictionary with the standard deviation of performance drift per shuffled
        feature.

    {features_to_drop_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {get_support}

    {transform}

    Notes
    -----
    This transformer is a similar concept to the `permutation_importance` from
    scikit-learn. The function in scikit-learn is used to evaluate feature importance
    instead of to select features.

    See Also
    --------
    sklearn.inspection.permutation_importance

    References
    ----------

    .. [1] Breiman, Random Forests, Machine Learning 45:5–32, 2001.

    Examples
    --------

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import SelectByShuffling
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> sbs = SelectByShuffling(
    >>>         RandomForestClassifier(random_state=42),
    >>>         cv=2,
    >>>         random_state=42,
    >>>       )
    >>> sbs.fit_transform(X, y)
       x2  x4  x5
    0   2   1   1
    1   4   2   1
    2   3   1   1
    3   1   1   1
    4   2   0   1
    5   2   1   1

    With polars:

    >>> import polars as pl
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import SelectByShuffling
    >>> X = pl.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pl.Series([1,0,0,1,1,0])
    >>> sbs = SelectByShuffling(
    >>>         RandomForestClassifier(random_state=42),
    >>>         cv=2,
    >>>         random_state=42,
    >>>       )
    >>> sbs.fit_transform(X, y)
    shape: (6, 3)
    ┌─────┬─────┬─────┐
    │ x2  ┆ x4  ┆ x5  │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 2   ┆ 1   ┆ 1   │
    │ 4   ┆ 2   ┆ 1   │
    │ 3   ┆ 1   ┆ 1   │
    │ 1   ┆ 1   ┆ 1   │
    │ 2   ┆ 0   ┆ 1   │
    │ 2   ┆ 1   ┆ 1   │
    └─────┴─────┴─────┘
    """

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        threshold: Union[float, int, None] = None,
        variables: Variables = None,
        random_state: Union[int, None] = None,
        confirm_variables: bool = False,
    ):

        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError(
                "threshold must be an integer, a float or None. "
                f"Got {threshold} instead."
            )

        super().__init__(confirm_variables)

        self.variables = _check_variables_input_value(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv
        self.random_state = random_state

    def fit(
        self,
        X: IntoDataFrame,
        y: IntoSeries,
        sample_weight: Union[MutableSequence, None] = None,
    ):
        """
        Find the important features.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
           The input dataframe. Can be a pandas, polars, or any other dataframe
           supported by narwhals.

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        """
        nw_X, y = check_X_y(X, y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.variables_ = _select_numerical_variables(
            X, self.variables, self.confirm_variables
        )

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv

        nw_X_model = nw_X.select(nw.col(*self.variables_))
        X_model = nw_X_model.to_native()

        # train model with all features and cross-validation
        model = cross_validate(
            estimator=self.estimator,
            X=X_model,
            y=y,
            cv=cv,
            return_estimator=True,
            return_indices=True,
            scoring=self.scoring,
            params={"sample_weight": sample_weight},
        )

        self.initial_model_performance_ = model["test_score"].mean()

        # the indices returned by cross_validate are the folds each model was
        # evaluated on, also when the splitter gives different folds on every call.
        validation_indices = model["indices"]["test"]
        y_val = [_safe_indexing(y, idx) for idx in validation_indices]

        scorer = get_scorer(self.scoring)
        random_state = check_random_state(self.random_state)
        n_samples = nw_X.shape[0]

        self.performance_drifts_ = {}
        self.performance_drifts_std_ = {}

        estimators = model["estimator"]

        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            X_val = [X_model.iloc[idx] for idx in validation_indices]
        else:
            nw_X_val = [nw_X_model[idx] for idx in validation_indices]

        for feature in self.variables_:
            # same permutation as pandas' sample(frac=1), so the drifts for a given
            # random_state are the same with every dataframe library.
            permutation = random_state.permutation(n_samples)

            # pandas is faster than narwhals.
            if nwd.is_pandas_dataframe(X) is True:
                values = X_model[feature].array
                performance = []
                for m, idx, X_, y_ in zip(estimators, validation_indices, X_val, y_val):
                    # the fold dataframes are copies made in fit, so we can shuffle
                    # the column in place and restore it afterwards.
                    original = X_[feature]
                    X_[feature] = values.take(permutation[idx])
                    performance.append(scorer(m, X_, y_))
                    X_[feature] = original
            else:
                column = nw_X_model.get_column(feature)
                performance = [
                    scorer(m, X_.with_columns(column[permutation[idx]]).to_native(), y_)
                    for m, idx, X_, y_ in zip(
                        estimators, validation_indices, nw_X_val, y_val
                    )
                ]

            # sklearn negates the error and loss scores, so larger is always better.
            drift = self.initial_model_performance_ - np.mean(performance)
            self.performance_drifts_[feature] = drift
            self.performance_drifts_std_[feature] = np.std(performance)

        # select features
        if not self.threshold:
            threshold = np.mean(list(self.performance_drifts_.values()))
        else:
            threshold = self.threshold

        self.features_to_drop_ = [
            f
            for f in self.performance_drifts_.keys()
            if self.performance_drifts_[f] < threshold
        ]

        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
