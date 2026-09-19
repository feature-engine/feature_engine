from types import GeneratorType
from typing import List, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame
from scipy.stats import kendalltau, rankdata
from sklearn.model_selection import cross_validate

from feature_engine.variable_handling import (
    check_all_variables,
    check_numerical_variables,
    find_all_variables,
    find_numerical_variables,
    retain_variables_if_in_df,
)

Variables = Union[int, str, List[Union[str, int]], None]


def get_feature_importances(estimator):
    """Retrieve feature importance from a fitted estimator"""

    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)

    if coef_ is not None:

        if estimator.coef_.ndim == 1:
            importances = np.abs(coef_)

        else:
            importances = np.linalg.norm(coef_, axis=0, ord=len(estimator.coef_))

        importances = list(importances)

    return importances


def _importance_series(X: IntoDataFrame, features, values: np.ndarray):
    """
    Return the importance of each feature as a pandas Series indexed by the
    features when X is a pandas dataframe, or as a dictionary with the features as
    keys otherwise.
    """
    if nwd.is_pandas_dataframe(X) is True:
        return nw.get_native_namespace(X).Series(values, index=features)
    return dict(zip(features, values.tolist()))


def _select_all_variables(
    X: IntoDataFrame,
    variables: Variables,
    confirm_variables: bool,
    exclude_datetime: bool = False,
):
    """
    Selects the variables over which the selector will operate.

    If variables is None, it will select all variables except datetime.
    If variables is a list and confirm_variables is True, it will retain those
    variables that are present in X. If confirm_variables is False, it will use all
    variables in the list.
    """
    if variables is None:
        variables_ = find_all_variables(X, exclude_datetime)
    else:
        if confirm_variables is True:
            variables_ = retain_variables_if_in_df(X, variables)
            variables_ = check_all_variables(X, variables_)
        else:
            variables_ = check_all_variables(X, variables)
    return variables_


def _select_numerical_variables(
    X: IntoDataFrame,
    variables: Variables,
    confirm_variables: bool,
):
    """
    Selects the numerical variables over which the selector will operate.

    If variables is None, it will select all numerical variables.
    If variables is a list and confirm_variables is True, it will retain those
    numerical variables that are present in X. If confirm_variables is False, it will
    use all numerical variables in the list.
    """
    if variables is None:
        variables_ = find_numerical_variables(X)
    else:
        if confirm_variables is True:
            variables_ = retain_variables_if_in_df(X, variables)
            variables_ = check_numerical_variables(X, variables_)
        else:
            variables_ = check_numerical_variables(X, variables)
    return variables_


def _corrcoef(values: np.ndarray) -> np.ndarray:
    # constant columns return NaN, like pandas, instead of warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.corrcoef(values, rowvar=False)


def _pearson_pairwise_complete(values: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """
    Pearson correlation of every pair of columns, using the rows where both are
    finite. The sums of all pairs come from matrix products, which is much faster
    than looping over the pairs.
    """
    mask: np.ndarray = finite.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # centring first keeps the sums below numerically stable.
        mean = np.where(finite, values, 0.0).sum(axis=0) / mask.sum(axis=0)
        x = np.where(finite, values - mean, 0.0)
        n_obs = mask.T @ mask
        sum_x = x.T @ mask
        sum_xx = (x * x).T @ mask
        var = sum_xx - sum_x * sum_x / n_obs
        corr = (x.T @ x - sum_x * sum_x.T / n_obs) / np.sqrt(var * var.T)

    # when the variance of the shared rows is tiny compared with the sums, the
    # subtraction above loses precision: recompute those pairs one by one.
    unstable = (var <= 1e-8 * sum_xx) | (var.T <= 1e-8 * sum_xx.T)
    corr[unstable] = np.nan
    for i, j in zip(*np.nonzero(np.triu(unstable & (n_obs > 1), 1))):
        rows = finite[:, i] & finite[:, j]
        corr[i, j] = _corrcoef(x[rows][:, [i, j]])[0, 1]
    return corr


def _spearman_pairwise_complete(nw_X: nw.DataFrame) -> np.ndarray:
    """
    Spearman correlation of every pair of columns, ranking each pair on the rows
    where both are finite, like pandas.DataFrame.corr().
    """
    variables = nw_X.columns
    exprs = []
    for i, var_i in enumerate(variables):
        for j in range(i + 1, len(variables)):
            var_j = variables[j]
            rows = nw.col(var_i).is_finite() & nw.col(var_j).is_finite()
            x = nw.when(rows).then(nw.col(var_i)).rank("average")
            y = nw.when(rows).then(nw.col(var_j)).rank("average")
            dx = x - x.mean()
            dy = y - y.mean()
            exprs.append(
                ((dx * dy).sum() / ((dx * dx).sum() * (dy * dy).sum()).sqrt()).alias(
                    f"__{i}_{j}__"
                )
            )
    n_vars = len(variables)
    corr = np.full((n_vars, n_vars), np.nan)
    corr[np.triu_indices(n_vars, 1)] = np.array(nw_X.select(exprs).row(0), dtype=float)
    return corr


def _correlation_matrix(X: IntoDataFrame, variables: list, method) -> np.ndarray:
    """
    Correlation matrix of the variables. Like pandas.DataFrame.corr(), each pair of
    variables is compared on the rows where both have finite values. Only the
    values above the diagonal are used.
    """
    if nwd.is_pandas_dataframe(X) is True:
        values = X[variables].to_numpy(dtype=float, na_value=np.nan)
    else:
        nw_X = nw.from_native(X, eager_only=True).select(nw.col(*variables))
        values = nw_X.to_numpy().astype(float)
    finite = np.isfinite(values)

    # numpy is faster than pandas and narwhals.
    if method == "pearson":
        if finite.all():
            return _corrcoef(values)
        return _pearson_pairwise_complete(values, finite)

    if method == "spearman" and finite.all():
        # scipy ranks faster than pandas, and polars faster than scipy.
        if nwd.is_pandas_dataframe(X) is True:
            return _corrcoef(rankdata(values, axis=0))
        return _corrcoef(nw_X.select(nw.all().rank("average")).to_numpy())

    # pandas is faster than narwhals.
    if nwd.is_pandas_dataframe(X) is True:
        return X[variables].corr(method=method).to_numpy()

    if method == "spearman":
        return _spearman_pairwise_complete(nw_X)

    # kendall and callables are computed pair by pair, as pandas does.
    corr_func = (lambda a, b: kendalltau(a, b)[0]) if method == "kendall" else method
    n_vars = len(variables)
    corr = np.full((n_vars, n_vars), np.nan)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            rows = finite[:, i] & finite[:, j]
            if rows.all():
                corr[i, j] = corr_func(values[:, i], values[:, j])
            elif rows.any():
                corr[i, j] = corr_func(values[rows, i], values[rows, j])
    return corr


def find_correlated_features(
    X: IntoDataFrame,
    variables: list[Union[str, int]],
    method: str,
    threshold: float,
):
    """
    Find groups of correlated variables.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The training dataset.

    variables : list
        The variables to examine.

    method: string or callable, default='pearson'
        Can take 'pearson', 'spearman', 'kendall' or callable. It refers to the
        correlation method to be used to identify the correlated features.

        - 'pearson': standard correlation coefficient
        - 'kendall': Kendall Tau correlation coefficient
        - 'spearman': Spearman rank correlation
        - callable: callable with input two 1d ndarrays and returning a float.

        For more details on this parameter visit the  `pandas.corr()` documentation.

    threshold: float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.

    Returns
    -------

    correlated_feature_groups: set
        Sets of correlated feature groups.

    features_to_drop: list
        The list of features that have been found to be correlated to at least one
        other feature.

    correlated_feature_dict: dict
        Dictionary containing the correlated feature groups. The key is the feature
        against which all other features were evaluated. The values are the features
        correlated with the key. The key + the values should be the same as the set
        found in `correlated_feature_groups`.
    """
    correlated_matrix = _correlation_matrix(X, variables, method)

    # the correlated pairs
    correlated_mask = np.triu(np.abs(correlated_matrix), 1) > threshold

    examined: np.ndarray = np.zeros(len(variables), dtype=bool)
    correlated_groups = list()
    features_to_drop = list()
    correlated_dict = {}
    for i, f_i in enumerate(variables):
        if examined.item(i) is False:
            examined[i] = True
            correlated = np.flatnonzero(correlated_mask[i] & ~examined)
            if len(correlated) > 0:
                examined[correlated] = True
                correlated_features = [variables[j] for j in correlated]
                features_to_drop.extend(correlated_features)
                correlated_groups.append({f_i, *correlated_features})
                correlated_dict[f_i] = set(correlated_features)

    return correlated_groups, features_to_drop, correlated_dict


def single_feature_performance(
    X: IntoDataFrame,
    y,
    variables: List[Union[str, int]],
    estimator,
    cv,
    scoring,
    groups=None,
):
    """
    Trains one estimator per feature and determines the performance of that estimator.

    Parameters
    ----------
    X: dataframe of shape = [n_samples, n_features]
       The input dataframe

    y: array-like of shape (n_samples)
       Target variable. Required to train the estimator.

    variables: list
        The variables to examine.

    estimator:
        Any scikit-learn estimator.

    cv:
        Cross-validation scheme. Any supported by the scikit-learn estimator.

    scoring:
        The performance metric. Any supported by the scikit-learn estimator.

    groups: Array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting
        the dataset into train/test set. Only used in conjunction with a
        “Group” cv instance (e.g., GroupKFold).

    Returns
    -------
    feature_performance: dict
        A dictionary with the feature name as key and the performance of the model
        trained with that feature as value.

    feature_performance_std: dict
        A dictionary with the feature name as key and the standard deviation of the
        performance of a model trained with that feature as value.
    """
    feature_performance = {}
    feature_performance_std = {}

    cv = list(cv) if isinstance(cv, GeneratorType) else cv
    nw_X = nw.from_native(X, eager_only=True)

    # train a model for every feature and store the performance
    for feature in variables:
        model = cross_validate(
            estimator,
            nw_X.get_column(feature).to_frame().to_native(),
            y,
            cv=cv,
            groups=groups,
            return_estimator=False,
            scoring=scoring,
        )

        feature_performance[feature] = model["test_score"].mean()
        feature_performance_std[feature] = model["test_score"].std()
    return feature_performance, feature_performance_std


def find_feature_importance(
    X: IntoDataFrame,
    y,
    estimator,
    cv,
    scoring,
    groups=None,
):
    """
    Trains an estimator using cross-validation and derives feature importance from it.
    The estimator needs to have the attributes `coef_` or `feature_importances_` after
    fitting. The importance is given by the coefficients of linear models or the purity
    gain obtained from tree-based models.

    Parameters
    ----------
    X: dataframe of shape = [n_samples, n_features]
       The input dataframe

    y: array-like of shape (n_samples)
       Target variable. Required to train the estimator.

    estimator:
        A scikit-learn estimator with parameters `coef_` or `feature_importances_`
        after fitting.

    cv:
        Cross-validation scheme. Any supported by the scikit-learn estimator.

    scoring:
        The performance metric. Any supported by the scikit-learn estimator.

    groups: Array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting
        the dataset into train/test set. Only used in conjunction with a
        “Group” cv instance (e.g., GroupKFold).

    Returns
    -------
    feature_importance: pandas Series or dict
        The importance of each feature, given by the coefficients of linear models or
        the impurity gain from tree-based models. A pandas Series with the feature
        names as index when X is a pandas dataframe, and a dictionary with the
        feature names as keys otherwise.

    feature_importance_std: pandas Series or dict
        The standard deviation of the importance of each feature, as a pandas Series
        or a dictionary, like `feature_importance`.
    """
    cv = list(cv) if isinstance(cv, GeneratorType) else cv

    model = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring=scoring,
        return_estimator=True,
    )

    importances = np.array([get_feature_importances(m) for m in model["estimator"]])

    # pandas keeps the columns index, with its name and dtype.
    if nwd.is_pandas_dataframe(X) is True:
        features = X.columns
    else:
        features = nw.from_native(X, eager_only=True).columns

    feature_importances_ = _importance_series(X, features, importances.mean(axis=0))
    feature_importances_std_ = _importance_series(
        X, features, importances.std(axis=0, ddof=1)
    )
    return feature_importances_, feature_importances_std_
