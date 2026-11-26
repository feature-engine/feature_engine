import datetime
import numbers
from typing import Dict, List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
import scipy.stats as stats
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
    _variables_attribute_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
)
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    find_categorical_and_numerical_variables,
    find_numerical_variables,
    retain_variables_if_in_df,
)

Variables = Union[None, int, str, List[Union[str, int]]]

PSI = """PSI = sum ( (test_i - basis_i) x ln(test_i/basis_i) )""".rstrip()


@Substitution(
    confirm_variables=_confirm_variables_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
    psi=PSI,
)
class DropHighPSIFeatures(BaseSelector):
    r"""
    DropHighPSIFeatures() drops features whose Population Stability Index (PSI) is
    above a given threshold.

    The PSI is used to compare distributions. Higher PSI values mean greater changes in
    a feature's distribution. Therefore, a feature with high PSI can be considered
    unstable.

    To compute the PSI, DropHighPSIFeatures() splits the dataset in two: a basis and
    a test set. Then, it compares the distribution of each feature between those sets.

    To determine the PSI, continuous features are sorted into discrete intervals, and
    then, the number of observations per interval are compared between the 2
    distributions.

    The PSI is calculated as:

    {psi}

    where `basis` and `test` are the 2 datasets, `i` refers to each interval, and then,
    `test_i` and `basis_i` are the number of observations in interval i in each data
    set.

    The PSI has traditionally been used to assess changes in distributions of
    continuous variables.

    In version 1.7, we extended the functionality of DropHighPSIFeatures() to
    calculate the PSI for categorical features as well. In this case, `i` is each
    unique category, and `test_i` and `basis_i` are the number of observations in
    category i.

    **Threshold**

    Different thresholds can be used to assess the magnitude of the distribution shift
    according to the PSI value. The most commonly used thresholds are:

    - Below 10%, the variable has not experienced a significant shift.
    - Above 25%, the variable has experienced a major shift.
    - Between those two values, the shift is intermediate.

    **Data split**

    To compute the PSI, DropHighPSIFeatures() splits the dataset in two: a basis and
    a test set. Then, it compares the distribution of each feature between those sets.

    There are various options to split a dataset:

    First, you can indicate which variable should be used to guide the data split. This
    variable can be of any data type. If you do not enter a variable name,
    DropHighPSIFeatures() will use the index of a pandas dataframe, or the row order of
    dataframes that don't have an index, like polars dataframes.

    Next, you need to specify how that variable (or the index) should be used to split
    the data. You can specify a proportion of observations to be put in each data set,
    or alternatively, provide a cut-off value.

    If you specify a proportion through the `split_frac` parameter, the data will
    be sorted to accommodate that proportion. If `split_frac` is 0.5, 50% of the
    observations will go to either basis or test sets. If `split_frac` is 0.6, 60% of
    the samples will go to the basis data set and the remaining 40% to the test set.

    If `split_distinct` is True, the data will be sorted considering unique values in
    the selected variables. Check the parameter below for more details.

    If you define a numeric cut-off value or a specific date using the `cut_off`
    parameter, the observations with value <= cut-off will go to the basis data set and
    the remaining ones to the test set. If the variable used to guide the split is
    categorical, its values are sorted alphabetically and cut accordingly.

    If you pass a list of values in the `cut-off`, the observations with the values in
    the list, will go to the basis set, and the remaining ones to the test set.

    More details in the :ref:`User Guide <psi_selection>`.

    Parameters
    ----------
    split_col: string or int, default=None.
        The variable that will be used to split the dataset into the basis and test
        sets. If None, the index of a pandas dataframe will be used. Dataframes without
        an index, like polars dataframes, are split by the position of the rows,
        from 0 to n_samples - 1, so the first rows go to the basis set. `split_col` can
        be a numerical, categorical or datetime variable. If `split_col` is a
        categorical variable, and the splitting criteria is given by `split_frac`, it
        will be assumed that the labels of the variable are sorted alphabetically.

    split_frac: float, default=0.5.
        The proportion of observations in each of the basis and test dataframes. If
        `split_frac` is 0.6, 60% of the observations will be put in the basis data set.

        If `split_distinct` is True, the indicated fraction may not be achieved exactly.
        See parameter `split_distinct` for more details.

        If `cut_off` is not None, `split_frac` will be ignored and the data split based
        on the `cut_off` value.

    split_distinct: boolean, default=False.
        If True, `split_frac` is applied to the vector of unique values in `split_col`
        instead of being applied to the whole vector of values. For example, if the
        values in `split_col` are [1, 1, 1, 1, 2, 2, 3, 4] and `split_frac` is
        0.5, we have the following:

            - `split_distinct=False` splits the vector in two equally sized parts:
                [1, 1, 1, 1] and [2, 2, 3, 4]. This involves that 2 dataframes with 4
                observations each are used for the PSI calculations.
            - `split_distinct=True` computes the vector of unique values in `split_col`
                ([1, 2, 3, 4]) and splits that vector in two equal parts: [1, 2] and
                [3, 4]. The number of observations in the two dataframes used for the
                PSI calculations is respectively 6 ([1, 1, 1, 1, 2, 2]) and 2 ([3, 4]).

    cut_off: int, float, date or list, default=None
        Threshold to split the dataset based on the `split_col` variable. If int, float
        or date, observations where the `split_col` values are <= threshold will
        go to the basis data set and the rest to the test set. If `cut_off` is a list,
        the observations where the `split_col` values are within the list will go to the
        basis data set and the remaining observations to the test set. If `cut_off` is
        not None, this parameter will be used to split the data and `split_frac` will be
        ignored.

    switch: boolean, default=False.
        If True, the order of the 2 dataframes used to determine the PSI (basis and
        test) will be switched. This is important because the interval limits used to
        calculate the PSI are inferred from the basis dataframe. Hence, changing the
        order of the dataframes may lead to different PSI values.

    threshold: float, str, default = 0.25.
        The threshold to drop a feature. If the PSI for a feature is >= threshold, the
        feature will be dropped. The most common threshold values are 0.25 (large shift)
        and 0.10 (medium shift).
        If 'auto', the threshold will be calculated based on the size of the basis and
        test dataset and the number of bins as:

                threshold = χ2(q, B−1) × (1/N + 1/M)

        where:

            - q = quantile of the distribution (or 1 - p-value),
            - B = number of bins/categories,
            - N = size of basis dataset,
            - M = size of test dataset.

        See formula (5.2) from reference [1].


    bins: int, default = 10
        Number of bins or intervals. For continuous features with good value spread, 10
        bins is commonly used. For features with lower cardinality or highly skewed
        distributions, lower values may be required.

    strategy: string, default='equal_frequency'
        If the intervals into which the features should be discretized are of equal
        size or equal number of observations. Takes values "equal_width" for equally
        spaced bins or "equal_frequency" for bins based on quantiles, that is, bins
        with similar number of observations.

    min_pct_empty_bins: float, default = 0.0001
        Value to add to empty bins or intervals. If after sorting the variable
        values into bins, a bin is empty, the PSI cannot be determined. By adding a
        small number to empty bins, we can avoid this issue. Note, that if the value
        added is too large, it may disturb the PSI calculation.

    missing_values: str, default='raise'
        Whether to perform the PSI feature selection on a dataframe with missing values.
        Takes values 'raise' or 'ignore'. If 'ignore', missing values will be dropped
        when determining the PSI for that particular feature. If 'raise' the transformer
        will raise an error and features will not be selected.

    p_value: float, default = 0.001
        The p-value to test the null hypothesis that there is no feature drift. In that
        case, the PSI-value approximates a random variable that follows a chi-square
        distribution. See [1] for details. This parameter is used only if `threshold`
        is set to 'auto'.

    variables: int, str, list, default = None
        The list of variables to evaluate. If `None`, the transformer will evaluate all
        numerical variables in the dataset. If `"all"` the transformer will evaluate all
        categorical and numerical variables in the dataset. Alternatively, the
        transformer will evaluate the variables indicated in the list or string.

    {confirm_variables}

    Attributes
    ----------
    features_to_drop_:
        List with the features that will be dropped.

    {variables_}

    psi_values_:
        Dictionary containing the PSI value per feature.

    cut_off_:
        Value used to split the dataframe into basis and test.
        This value is computed when not given as parameter. When `split_col` is None
        and the dataframe has no index, it refers to the position of the rows.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find features with high PSI values.

    {fit_transform}

    {get_support}

    transform:
        Remove features with high PSI values.

    See Also
    --------
    feature_engine.discretisation.EqualFrequencyDiscretiser
    feature_engine.discretisation.EqualWidthDiscretiser

    References
    ----------
    .. [1] Yurdakul B. "Statistical properties of population stability index".
       Western Michigan University, 2018.
       https://scholarworks.wmich.edu/dissertations/3208/

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import DropHighPSIFeatures
    >>> X = pd.DataFrame(dict(
    >>>         x1 = [1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    >>>         x2 = [32,87,6,32,11,44,8,7,9,0,32,87,6,32,11,44,8,7,9,0],
    >>>         ))
    >>> psi = DropHighPSIFeatures()
    >>> psi.fit_transform(X)
        x2
    0   32
    1   87
    2    6
    3   32
    4   11
    5   44
    6    8
    7    7
    8    9
    9    0
    10  32
    11  87
    12   6
    13  32
    14  11
    15  44
    16   8
    17   7
    18   9
    19   0

    With a polars dataframe, which has no index, the rows are split by their
    position:

    >>> import polars as pl
    >>> X = pl.DataFrame(dict(
    >>>         x1 = [1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    >>>         x2 = [32,87,6,32,11,44,8,7,9,0,32,87,6,32,11,44,8,7,9,0],
    >>>         ))
    >>> psi = DropHighPSIFeatures()
    >>> psi.fit_transform(X).columns
    ['x2']
    """

    def __init__(
        self,
        split_col: Union[str, None] = None,
        split_frac: float = 0.5,
        split_distinct: bool = False,
        cut_off: Union[None, int, float, datetime.date, List] = None,
        switch: bool = False,
        threshold: Union[float, int, str] = 0.25,
        bins: int = 10,
        strategy: str = "equal_frequency",
        min_pct_empty_bins: float = 0.0001,
        missing_values: str = "raise",
        variables: Variables = None,
        confirm_variables: bool = False,
        p_value: float = 0.001,
    ):

        if not isinstance(split_col, (str, int, type(None))):
            raise ValueError(
                f"split_col must be a string an integer or None. Got "
                f"{split_col} instead."
            )

        if split_frac is None and cut_off is None:
            raise ValueError(
                f"cut_off and split_frac cannot be both set to None. "
                f"The current values are {split_frac, cut_off}. Please "
                f"specify a value for at least one of these parameters."
            )

        # split_frac is only used when cut_off is None.
        if cut_off is None:
            if not isinstance(split_frac, (float, int)) or not 0 < split_frac < 1:
                raise ValueError(
                    f"split_frac must be a float between 0 and 1. Got {split_frac} "
                    f"instead."
                )

        if not isinstance(split_distinct, bool):
            raise ValueError(
                f"split_distinct must be a boolean. Got {split_distinct} instead."
            )

        if not isinstance(switch, bool):
            raise ValueError(f"switch must be a boolean. Got {switch} instead.")

        if not (
            (isinstance(threshold, str) and threshold == "auto")
            or (isinstance(threshold, numbers.Real) and threshold >= 0)
        ):
            raise ValueError(
                f"threshold must be greater than 0 or 'auto'. Got {threshold} instead."
            )

        if not isinstance(bins, int) or bins <= 1:
            raise ValueError(f"bins must be an integer >= 2. Got {bins} instead.")

        if not isinstance(strategy, str) or strategy not in [
            "equal_width",
            "equal_frequency",
        ]:
            raise ValueError(
                "strategy takes only values equal_width or equal_frequency. Got "
                f"{strategy} instead."
            )

        if not isinstance(min_pct_empty_bins, (float, int)) or min_pct_empty_bins < 0:
            raise ValueError(
                f"min_pct_empty_bins must be >= 0. Got {min_pct_empty_bins} "
                f"instead."
            )

        if not isinstance(missing_values, str) or missing_values not in [
            "raise",
            "ignore",
        ]:
            raise ValueError(
                f"missing_values takes only values 'raise' or 'ignore'. Got "
                f"{missing_values} instead."
            )

        if isinstance(variables, list):
            if split_col in variables:
                raise ValueError(
                    f"{split_col} cannot be used to split the data and be evaluated at "
                    f"the same time. Either remove {split_col} from the variables list "
                    f"or choose another splitting criteria."
                )

        if not isinstance(p_value, float) or p_value < 0 or p_value > 1:
            raise ValueError(
                f"p_value must be a float between 0 and 1. Got {p_value} instead."
            )

        super().__init__(confirm_variables)

        # Check the variables before assignment.
        self.variables = _check_variables_input_value(variables)

        # Set all remaining arguments as attributes.
        self.split_col = split_col
        self.split_frac = split_frac
        self.split_distinct = split_distinct
        self.cut_off = cut_off
        self.switch = switch
        self.threshold = threshold
        self.bins = bins
        self.strategy = strategy
        self.min_pct_empty_bins = min_pct_empty_bins
        self.missing_values = missing_values
        self.p_value = p_value

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Find features with high PSI values.

        Parameters
        ----------
        X : dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : Series. Default = None
            y is not needed in this transformer. You can pass y or None.
        """
        nw_X = check_X(X)

        cat_variables_, num_variables_ = self._select_variables(X)

        # the split column is removed from the variables selected automatically
        cat_variables_, num_variables_ = self._check_split_column(
            nw_X, cat_variables_, num_variables_
        )

        if self.missing_values == "raise":
            _check_contains_na(X, num_variables_ + cat_variables_)
        # the intervals can't be computed with inf values, even if NaN are ignored
        _check_contains_inf(X, num_variables_)

        is_basis = self._basis_mask(X, nw_X)
        n_basis = int(is_basis.sum())
        n_test = is_basis.shape[0] - n_basis

        if min(n_basis, n_test) < self.bins:
            raise ValueError(
                "The number of rows in the basis and test datasets that will be used "
                f"in the PSI calculations must be at least larger than {self.bins}. "
                "After splitting the original dataset based on the given cut_off or "
                f"split_frac we have {n_basis} samples in the basis set, "
                f"and {n_test} samples in the test set. "
                "Please adjust the value of the cut_off or split_frac."
            )

        if self.switch is True:
            is_basis = ~is_basis
            n_basis, n_test = n_test, n_basis

        if self.threshold == "auto":
            threshold_num = self._calculate_auto_threshold(n_basis, n_test, self.bins)
        else:
            threshold_num = self.threshold

        self.psi_values_: Dict = {}
        self.features_to_drop_ = []

        for feature in num_variables_:
            # pandas is faster than narwhals.
            if nwd.is_pandas_dataframe(X) is True:
                values = X[feature].to_numpy()
            else:
                values = nw_X.get_column(feature).to_numpy()
            basis = values[is_basis]
            test = values[~is_basis]
            if self.missing_values == "ignore":
                basis = basis[~np.isnan(basis)]
                test = test[~np.isnan(test)]
                self._check_observations(feature, basis, test)

            # the intervals are learned from the basis set only
            limits = self._interval_limits(basis)
            n_intervals = len(limits) + 1
            basis_counts = np.bincount(
                np.searchsorted(limits, basis, side="left"), minlength=n_intervals
            )
            test_counts = np.bincount(
                np.searchsorted(limits, test, side="left"), minlength=n_intervals
            )
            self.psi_values_[feature] = self._psi(basis_counts, test_counts)

            if self.psi_values_[feature] > threshold_num:
                self.features_to_drop_.append(feature)

        for feature in cat_variables_:
            basis_counts, test_counts = self._category_counts(
                X, nw_X, feature, is_basis
            )
            self.psi_values_[feature] = self._psi(basis_counts, test_counts)

            if self.threshold == "auto":
                n_categories = np.count_nonzero(basis_counts + test_counts)
                threshold_cat = self._calculate_auto_threshold(
                    n_basis, n_test, n_categories
                )
            else:
                threshold_cat = self.threshold

            if self.psi_values_[feature] > threshold_cat:
                self.features_to_drop_.append(feature)

        self.variables_ = num_variables_ + cat_variables_
        self._get_feature_names_in(X)

        return self

    def _select_variables(self, X: IntoDataFrame):
        """Based on the user input to the `variables` attribute in init, find the
        numerical and categorical variables for which the PSI should be calculated.

        If `None`, select all numerical variables.
        If `"all"`, select all numerical and categorical variables.
        If string, int, or list, split into lists of numerical or categorical variables.
        """

        if self.variables is None:
            num_variables = find_numerical_variables(X)
            cat_variables: List[Union[str, int]] = []

        elif self.variables == "all":
            (
                cat_variables,
                num_variables,
            ) = find_categorical_and_numerical_variables(X, None)

        else:
            if self.confirm_variables is True:
                variables = retain_variables_if_in_df(X, self.variables)
            else:
                variables = self.variables

            (
                cat_variables,
                num_variables,
            ) = find_categorical_and_numerical_variables(X, variables)

        return cat_variables, num_variables

    def _check_split_column(
        self,
        nw_X: nw.DataFrame,
        cat_variables: List[Union[str, int]],
        num_variables: List[Union[str, int]],
    ):
        """Check that split_col is in the dataframe and remove from numerical and
        categorical variable lists if necessary.

        It will get added if the variables are selected automatically.
        """
        if self.split_col is not None:
            if self.split_col not in nw_X.columns:
                raise ValueError(f"{self.split_col} is not in the dataframe.")

            if self.variables is None or self.variables == "all":
                if self.split_col in num_variables:
                    num_variables.remove(self.split_col)
                elif self.split_col in cat_variables:
                    cat_variables.remove(self.split_col)

        return cat_variables, num_variables

    def _check_observations(self, feature, basis: np.ndarray, test: np.ndarray):
        """Check that the basis and test sets have values of the feature, which can
        be missing when NaN are ignored."""
        if len(basis) == 0 or len(test) == 0:
            raise ValueError(
                f"The variable {feature} has only missing values in the basis or in "
                "the test set, so its PSI can't be computed. Got "
                f"{len(basis)} values in the basis set and {len(test)} values in the "
                "test set."
            )

    def _interval_limits(self, values: np.ndarray) -> np.ndarray:
        """Inner limits of the intervals, the same ones EqualFrequencyDiscretiser
        and EqualWidthDiscretiser find. The first and last intervals are open, so
        values outside the basis range go to them."""
        if self.strategy == "equal_frequency":
            quantiles = np.linspace(0, 1, self.bins + 1)
            # round up quantiles that are not exact in base 2, as pandas.qcut does
            np.putmask(
                quantiles,
                self.bins * quantiles != np.arange(self.bins + 1),
                np.nextafter(quantiles, 1),
            )
            limits = np.unique(np.quantile(values, quantiles, method="linear"))
        else:
            low, high = np.min(values), np.max(values)
            # widen a constant range by 0.1%, as pandas.cut does
            if low == high:
                low = low - 0.001 * abs(low) if low != 0 else -0.001
                high = high + 0.001 * abs(high) if high != 0 else 0.001
            limits = np.unique(np.linspace(low, high, self.bins + 1))
        return limits[1:-1]

    def _category_counts(
        self, X: IntoDataFrame, nw_X: nw.DataFrame, feature, is_basis: np.ndarray
    ):
        """Number of observations per category in the basis and test sets, with
        the categories sorted. Missing values are not counted."""
        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            codes, categories = X[feature].factorize(sort=True)
            is_value = codes >= 0
            basis_counts = np.bincount(
                codes[is_basis & is_value], minlength=len(categories)
            )
            test_counts = np.bincount(
                codes[~is_basis & is_value], minlength=len(categories)
            )
        else:
            counts = (
                nw_X.select(nw.col(feature))
                .with_columns(
                    nw.new_series(
                        "__basis__",
                        is_basis,
                        backend=nw.get_native_namespace(nw_X),
                    )
                )
                .drop_nulls()
                .group_by(feature)
                .agg(nw.col("__basis__").sum(), nw.len().alias("__count__"))
                .sort(feature)
            )
            basis_counts = counts.get_column("__basis__").to_numpy()
            test_counts = counts.get_column("__count__").to_numpy() - basis_counts
        return basis_counts, test_counts

    def _psi(self, basis_counts: np.ndarray, test_counts: np.ndarray) -> float:
        """PSI from the number of observations per interval or category."""
        # intervals or categories that are empty in both sets don't add to the PSI
        observed = (basis_counts > 0) | (test_counts > 0)
        basis = basis_counts[observed] / basis_counts.sum()
        test = test_counts[observed] / test_counts.sum()
        basis[basis == 0] = self.min_pct_empty_bins
        test[test == 0] = self.min_pct_empty_bins
        # min_pct_empty_bins=0 gives an inf PSI with empty intervals, without warnings
        with np.errstate(divide="ignore"):
            return np.sum((test - basis) * np.log(test / basis))

    def _basis_mask(self, X: IntoDataFrame, nw_X: nw.DataFrame) -> np.ndarray:
        """
        Find the observations of the basis dataset.

        The basis dataset contains the observations whose value in `split_col` (or
        in the index, or the row position when the dataframe has no index) is <=
        cut_off, or is in cut_off when it is a list. The test dataset contains the
        remaining observations.

        Parameters
        ----------
        X : dataframe

        nw_X : narwhals dataframe
            X in narwhals format.

        Returns
        -------
        is_basis: numpy array
            Boolean array that is True for the observations of the basis dataset.
        """
        # pandas is faster than narwhals, and has an index.
        if nwd.is_pandas_dataframe(X) is True:
            if self.split_col is None:
                reference = X.index.to_series()
            else:
                reference = X[self.split_col]
            n_missing = reference.isna().sum()
        else:
            if self.split_col is None:
                # without an index, the row order is the reference
                reference = nw.new_series(
                    "__row__",
                    np.arange(nw_X.shape[0]),
                    backend=nw.get_native_namespace(nw_X),
                )
            else:
                reference = nw_X.get_column(self.split_col)
            n_missing = reference.null_count()
            if reference.dtype.is_float() is True:
                n_missing += reference.is_nan().sum()

        if n_missing > 0:
            raise ValueError(
                f"There are {n_missing} missing values in the reference "
                "variable. Missing data are not allowed in the variable used to "
                "split the dataframe."
            )

        if self.cut_off is None:
            self.cut_off_ = self._get_cut_off_value(X, reference)
        else:
            self.cut_off_ = self.cut_off

        if isinstance(self.cut_off_, list):
            cut_off = self.cut_off_
            if nwd.is_pandas_dataframe(X) is True:
                # isin with dates or strings is deprecated with datetime columns.
                if reference.dtype.kind == "M":
                    cut_off = np.array(cut_off, dtype="datetime64[ns]")
                is_basis = reference.isin(cut_off)
            else:
                # polars can't compare dates with datetimes.
                if reference.dtype == nw.Datetime:
                    cut_off = (
                        nw.new_series(
                            "__cut_off__",
                            cut_off,
                            backend=nw.get_native_namespace(nw_X),
                        )
                        .cast(reference.dtype)
                        .to_list()
                    )
                is_basis = reference.is_in(cut_off)
        else:
            is_basis = reference <= self.cut_off_

        return is_basis.to_numpy()

    def _get_cut_off_value(self, X: IntoDataFrame, split_column):
        """
        Find the cut-off value to split the dataframe. It is implemented when the user
        does not enter a cut_off value as a parameter. It is calculated based on
        split_frac.

        Finds the value in a series at which we find the split_frac percentage
        of observations.

        If the reference column is numerical, the cut-off value is determined using
        np.quantile. Otherwise, the cut-off value is based on the value_counts:

            - The distinct values are sorted and the cumulative sum is
            used to compute the quantile. The value with the quantile that
            is the closest to the chosen split fraction is used as cut-off.

            - The procedure assumes that categorical values are sorted alphabetically
            and cut accordingly.

        Parameters
        ----------
        X: dataframe
            The dataframe to split.

        split_column: pandas or narwhals series.
            Series for which the nth quantile will be computed: a pandas series if X
            is a pandas dataframe, or a narwhals series otherwise.

        Returns
        -------
        cut_off: (float, int, str, object).
            value for the cut-off.
        """
        if nwd.is_pandas_dataframe(X) is True:
            if self.split_distinct is True:
                split_column = split_column.drop_duplicates()

            if split_column.dtype.kind in "biufc":
                cut_off = np.quantile(split_column, self.split_frac)
            else:
                cumulative = (
                    split_column.value_counts(normalize=True).sort_index().cumsum()
                )
                # the value whose cumulative fraction is the closest to split_frac
                position = np.argmin(np.abs(cumulative.to_numpy() - self.split_frac))
                cut_off = cumulative.index.to_numpy()[position]
        else:
            if self.split_distinct is True:
                split_column = split_column.unique()

            if split_column.dtype.is_numeric() is True:
                cut_off = np.quantile(split_column.to_numpy(), self.split_frac)
            else:
                proportions = split_column.value_counts(
                    name="__proportion__", normalize=True
                ).sort(split_column.name)
                cumulative = proportions.get_column("__proportion__").cum_sum()
                position = np.argmin(np.abs(cumulative.to_numpy() - self.split_frac))
                cut_off = proportions.get_column(split_column.name).item(int(position))

        return cut_off

    def _calculate_auto_threshold(self, N, M, bins):
        """Threshold computation for chi-square test.

        The threshold is given by:

            threshold = χ2(q,B−1) × (1/N + 1/M)

        where:

        q = quantile of the distribution (or 1 - p-value),
        B = number of bins/categories,
        N = size of basis dataset,
        M = size of test dataset.
        See formula (5.2) from reference [1] in the class docstring.

        Parameters
        ----------
        N: float or int
        M: float or int
        bins: int

        Returns
        -------
        float
        """
        return stats.chi2.ppf(1 - self.p_value, bins - 1) * (1.0 / N + 1.0 / M)

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "pass"
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
