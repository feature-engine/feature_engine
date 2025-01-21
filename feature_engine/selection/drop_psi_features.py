import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from pandas.api.types import is_numeric_dtype

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
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
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
    DropHighPSIFeatures() drops features which Population Stability Index (PSI) is
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
    DropHighPSIFeatures() will use the dataframe index.

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
        sets. If None, the dataframe index will be used. `split_col` can be a numerical,
        categorical or datetime variable. If `split_col` is a categorical variable, and
        the splitting criteria is given by `split_frac`, it will be assumed that the
        labels of the variable are sorted alphabetically.

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
        test) will be switched. This is important because the PSI is not symmetric,
        i.e., PSI(a, b) != PSI(b, a)).

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
        This value is computed when not given as parameter.

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

        # split_frac and cut_off can't be None at the same time
        if not split_frac and not cut_off:
            raise ValueError(
                f"cut_off and split_frac cannot be both set to None "
                f"The current values are {split_frac, cut_off}. Please "
                f"specify a value for at least one of these parameters."
            )

        # check split_frac only if it will be used.
        if split_frac and not cut_off:
            if not (0 < split_frac < 1):
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

        if (isinstance(threshold, str) and (threshold != "auto")) or (
            isinstance(threshold, (float, int)) and threshold < 0
        ):
            raise ValueError(
                f"threshold must be greater than 0 or 'auto'. Got {threshold} instead."
            )

        if not isinstance(bins, int) or bins <= 1:
            raise ValueError(f"bins must be an integer >= 1. Got {bins} instead.")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "strategy takes only values equal_width or equal_frequency. Got "
                f"{strategy} instead."
            )

        if not isinstance(min_pct_empty_bins, (float, int)) or min_pct_empty_bins < 0:
            raise ValueError(
                f"min_pct_empty_bins must be >= 0. Got {min_pct_empty_bins} "
                f"instead."
            )

        if missing_values not in ["raise", "ignore"]:
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

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Find features with high PSI values.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : pandas series. Default = None
            y is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = check_X(X)

        # select variables to evaluate
        cat_variables_, num_variables_ = self._select_variables(X)

        # check that split column is in the dataframe and remove from variable lists
        cat_variables_, num_variables_ = self._check_split_column(
            X, cat_variables_, num_variables_
        )

        if self.missing_values == "raise":
            # check if dataset contains na or inf
            _check_contains_na(X, num_variables_ + cat_variables_)
            _check_contains_inf(X, num_variables_)

        # Split the dataframe into basis and test.
        basis_df, test_df = self._split_dataframe(X)

        # Check the shape of the returned dataframes for PSI calculations.
        # The number of observations must be at least equal to the
        # number of bins.
        if min(basis_df.shape[0], test_df.shape[0]) < self.bins:
            raise ValueError(
                "The number of rows in the basis and test datasets that will be used "
                f"in the PSI calculations must be at least larger than {self.bins}. "
                "After slitting the original dataset based on the given cut_off or"
                f"split_frac we have {basis_df.shape[0]} samples in the basis set, "
                f"and {test_df.shape[0]} samples in the test set. "
                "Please adjust the value of the cut_off or split_frac."
            )

        # Switch basis and test dataframes if required.
        if self.switch:
            test_df, basis_df = basis_df, test_df

        # Set up parameters for numerical features
        if len(num_variables_) > 0:

            # Set up the discretizer for numerical features
            if self.strategy == "equal_width":
                bucketer = EqualWidthDiscretiser(bins=self.bins)
            else:
                bucketer = EqualFrequencyDiscretiser(q=self.bins)

            # Set up the threshold for numerical features
            if self.threshold == "auto":
                threshold_num = self._calculate_auto_threshold(
                    basis_df.shape[0],
                    test_df.shape[0],
                    self.bins,
                )
            else:
                threshold_num = self.threshold

        # Set up the generic threshold for categorical features if used
        if len(cat_variables_) > 0:
            if self.threshold != "auto":
                threshold_cat = self.threshold

        # Compute the PSI by looping over the features
        self.psi_values_: Dict = {}
        self.features_to_drop_ = []

        # Compute PSI for numerical features
        for feature in num_variables_:
            # Discretize feature
            basis_discrete = bucketer.fit_transform(basis_df[[feature]].dropna())
            test_discrete = bucketer.transform(test_df[[feature]].dropna())

            # Determine percentage of observations per bin
            basis_distrib, test_distrib = self._observation_frequency_per_bin(
                basis_discrete, test_discrete
            )

            # Calculate the PSI value
            self.psi_values_[feature] = np.sum(
                (test_distrib - basis_distrib) * np.log(test_distrib / basis_distrib)
            )

            # Assess if feature should be dropped
            if self.psi_values_[feature] > threshold_num:
                self.features_to_drop_.append(feature)

        # Compute the PSI for categorical features
        for feature in cat_variables_:
            basis_discrete = basis_df[[feature]]
            test_discrete = test_df[[feature]]

            # Determine percentage of observations per bin
            basis_distrib, test_distrib = self._observation_frequency_per_bin(
                basis_discrete, test_discrete
            )

            # Calculate the PSI value
            self.psi_values_[feature] = np.sum(
                (test_distrib - basis_distrib) * np.log(test_distrib / basis_distrib)
            )

            # Determine the appropriate threshold for the categorical feature
            if self.threshold == "auto":
                n_bins_cat = X[feature].nunique()
                threshold_cat = self._calculate_auto_threshold(
                    basis_df.shape[0],
                    test_df.shape[0],
                    n_bins_cat,
                )

            # Assess if feature should be dropped
            if self.psi_values_[feature] > threshold_cat:
                self.features_to_drop_.append(feature)

        # store analyzed variables
        self.variables_ = num_variables_ + cat_variables_
        # save input features
        self._get_feature_names_in(X)

        return self

    def _select_variables(self, X: pd.DataFrame):
        """Based on the user input to the `variables` attribute in init, find the
        numerical and categorical variables for which the PSI should be calculated.

        If `None`, select all numerical variables.
        If `"all", select all numerical and categorical variables.
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
        X: pd.DataFrame,
        cat_variables: List[Union[str, int]],
        num_variables: List[Union[str, int]],
    ):
        """Check that split_col is in the dataframe and remove from numerical and
        categorical variable lists if necessary.

        It will get added if the variables are selected automatically.
        """
        if self.split_col is not None:
            # check that split_col is in the dataframe.
            if self.split_col not in X.columns:
                raise ValueError(f"{self.split_col} is not in the dataframe.")

            # Remove the split_col from variables lists. Happens when variables are
            # selected automatically by the transformer.
            if self.variables is None or self.variables == "all":
                if self.split_col in num_variables:
                    num_variables.remove(self.split_col)
                elif self.split_col in cat_variables:
                    cat_variables.remove(self.split_col)

        return cat_variables, num_variables

    def _observation_frequency_per_bin(self, basis, test):
        """
        Obtain the fraction of observations per interval.

        Parameters
        ----------
        basis : pd.DataFrame.
            The basis Pandas DataFrame with discretised (i.e., binned) values.

        test: pd.DataFrame.
            The test Pandas DataFrame with discretised (i.e., binned) values.

        Returns
        -------
        distribution.basis: pd.Series.
            Basis Pandas Series with percentage of observations per bin.

        distribution.meas: pd.Series.
            Test Pandas Series with percentage of observations per bin.
        """
        # Compute the feature distribution for basis and test
        basis_distrib = basis.value_counts(normalize=True)
        test_distrib = test.value_counts(normalize=True)

        # Align the two distributions by merging the buckets (bins). This ensures
        # the number of bins is the same for the two distributions (in case of
        # empty buckets).
        distributions = (
            pd.DataFrame(basis_distrib)
            .merge(
                pd.DataFrame(test_distrib),
                right_index=True,
                left_index=True,
                how="outer",
            )
            .fillna(self.min_pct_empty_bins)
            .replace(to_replace=0, value=self.min_pct_empty_bins)
        )
        distributions.columns = ["basis", "test"]

        return distributions.basis, distributions.test

    def _split_dataframe(self, X: pd.DataFrame):
        """
        Split dataframe according to a cut-off value and return two dataframes: the
        basis dataframe contains all observations <= cut_off and the test dataframe the
        observations > cut_off.

        If cut-off is a list, then the basis dataframe will contain all observations
        which values are within the list, and the test dataframe all remaining
        observations.

        The cut-off value is associated to a specific column.

        Parameters
        ----------
        X : pandas dataframe

        Returns
        -------
        basis_df: pd.DataFrame
            pandas dataframe with observations which value <= cut_off

        test_df: pd.DataFrame
            pandas dataframe with observations which value > cut_off
        """

        # Identify the values according to which the split must be done.
        if self.split_col is None:
            reference = pd.Series(X.index)
        else:
            reference = X[self.split_col]

        # Raise an error if there are missing values in the reference column.
        if reference.isna().any():
            raise ValueError(
                f"There are {reference.isna().sum()} missing values in the reference"
                "variable. Missing data are not allowed in the variable used to "
                "split the dataframe."
            )

        # If cut_off is not pre-defined, compute it.
        if not self.cut_off:
            self.cut_off_ = self._get_cut_off_value(reference)
        else:
            self.cut_off_ = self.cut_off

        # Split the original dataframe
        if isinstance(self.cut_off_, list):
            is_within_cut_off = np.array(reference.isin(self.cut_off_))

        else:
            is_within_cut_off = np.array(reference <= self.cut_off_)

        basis_df = X[is_within_cut_off]
        test_df = X[~is_within_cut_off]

        return basis_df, test_df

    def _get_cut_off_value(self, split_column):
        """
        Find the cut-off value to split the dataframe. It is implemented when the user
        does not enter a cut_off value as a parameter. It is calculated based on
        split_frac.

        Finds the value in a pandas series at which we find the split_frac percentage
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
        split_column: pd.Series.
            Series for which the nth quantile will be computed.

        Returns
        -------
        cut_off: (float, int, str, object).
            value for the cut-off.
        """

        # In case split_distinct is used, extract series with unique values
        if self.split_distinct:
            split_column = pd.Series(split_column.unique())

        # If the value is numerical, use numpy functionality
        if is_numeric_dtype(split_column):
            cut_off = np.quantile(split_column, self.split_frac)

        # Otherwise use value_counts combined with cumsum
        else:
            reference = pd.DataFrame(
                split_column.value_counts(normalize=True).sort_index().cumsum()
            )

            # Get the index (i.e. value) with the quantile that is the closest
            # to the split_frac defined at initialization.
            distance = abs(reference - self.split_frac)
            cut_off = (distance.idxmin()).values[0]

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
