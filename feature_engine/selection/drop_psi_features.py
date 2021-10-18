from typing import List, Union

import datetime
import numpy as np
import pandas as pd

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _is_dataframe,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


class DropHighPSIFeatures(BaseSelector):
    r"""
    DropHighPSIFeatures drops features with a Population Stability Index (PSI) value
    above a given threshold. The PSI of a numerical feature is an indication of the
    shift in its distribution; a feature with high PSI might therefore be considered
    unstable.

    To compute the PSI the transformer splits the dataset in two: a base and a test set.
    The base set is assumed to contain the expected feature distributions. The test set
    will be assessed against the base set.

    In Credit Risk, eliminating features with high PSI is commonly done and
    usually required by the Regulator.

    To determine the PSI, continuous features are sorted into discrete intervals, the
    number of observations per intervals is then determined, and those values are
    compared between the base and test sets.

    In other words, the PSI is computed as follows:

    - Define the bins into which the observations will be sorted using the base set.
    - Sort the feature values into those bins (for both base and test sets).
    - Determine the fraction of observations within each bin.
    - Compute the PSI.

    The PSI is determined as:

    PSI = \sum_{i=1}^n (test_i - base_i) . ln(\frac{test_i}{base_i})

    A bigger value of the PSI indicates a bigger shift in the feature distribution
    between the base and test sets.

    Different thresholds can be used to assess the importance of the population shift
    reported by the PSI value. The most commonly used thresholds are:

    - Below 10%, the variable has not experienced a significant shift.
    - Above 25%, the variable has experienced a major shift.
    - Between those two values, the shift is intermediate.

    When working with PSI, it is worth highlighting the following:

    - The PSI is not symmetric; switching the order of the dataframes used as base and
    test will lead to different PSI values.
    - The number of bins has an impact on the PSI values.
    - The PSI is a suitable metric for numerical features (i.e., either continuous or
    with high cardinality). For categorical or discrete features, the change in
    distributions is better assessed with Chi-squared.

    To compute the PSI the DropHighPSIFeatures splits the dataset in two. To split the
    dataframe, several options are available.

    First and foremost, the user can enter one variable, which will be used to guide the
    data split. This variable can take numbers, strings or dates as values. If the user
    does not enter a variable name, DropHighPSIFeatures will use the dataframe index.

    Second, the user has the option to specify a proportion of observations to put in
    each data set, or alternatively, provide a cut-off value.

    If the user specifies a proportion through the split_frac parameter, the data will
    be sorted to accommodate that proportion. If split_frac is 0.5, 50% of the
    observations will go to either base or test sets. If split_frac is 0.6, 60% of the
    samples will go to the base test and the remaining 40% to the test set.

    If the user defines a numeric cut-off value or a specific date using the cut_off
    parameter, the observations with value below the cut-off will go to the base set and
    the remaining ones to the test set.

    If the user passes a list of values, the observations with the values in the list,
    will go to the base set, alternatively to the test set.


    References
    ----------
    https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations


    Parameters
    ----------

    split_col: string or int, default=None.
        The variable that will be used to split the dataset. If None, the dataframe
        index will be used.

    split_frac: float, default=0.5.
        The proportion of observations in each of the dataframes that will be used
        to compare the feature distributions. If split_distinct is True, the indicated
        ratio may not be achieved exactly. See parameter split_distinct for more
        details. If cut_off is not None, split_frac will be ignored and the data split
        based on the cut_off value.

    split_distinct: boolean, default=False.
        If True, split_col unique values will go to either dataframe but not both. For
        example, if split_col is [0, 1, 1, 1, 2, 2], split_frac is 0.5 and
        split_distinct is False, the data will be divided ind [0, 1, 1] and [1, 2, 2]
        achieving exactly a 50% split. However, if split_distinct is True, then the data
        will be divided into [0, 1, 1, 1] and [2, 2], with an approximate split of 0.5
        but not exactly.

    cut_off: None, int, float, date or list, default=None
        Threshold to split the dataset based on the split_col values. If int, float
        or date, observations where the split_col values are below the threshold will
        go to the base dataframe and those with values above the threshold will land in
        the test dataframe. If cut_off is a list, the observations where the split_col
        values are within the list will go to the base dataframe and the remaining
        observations will land in the test dataframe. If cut_off is not None, this
        parameter will be used to split the data and split_frac will be ignored.

    switch: boolean, default=False.
        If True, the order of the 2 dataframes used to determine the PSI (base and
        test) will be switched. This is important because the PSI is not symmetric,
        i.e., PSI(a, b) != PSI(b, a)).

    threshold: float, default = 0.25.
        The threshold to drop a feature. If the PSI for a feature is >= threshold, the
        feature will be dropped. The most common threshold values are 0.25 (large shift)
        and 0.10 (medium shift).

    bins: int, default = 10
        Number of bins or intervals. For continuous features with good value spread, 10
        bins is commonly used. For features with lower cardinality or highly skewed
        distributions, lower values may be required.

    strategy: string, default='equal_frequency'
        Type of binning used to represent the distribution of the feature. In can be
        "equal_width" for equally spaced bins or "equal_frequency" for bins based on
        quantiles, that is, bins with similar number of observations.

    min_pct_empty_buckets: float, default = 0.0001
        Value to add to empty buckets or empty bins. If after sorting the variable
        values into bins, a bin is empty the PSI cannot be determined. By adding a
        small number to empty bins, we can avoid this issue. Note, that if the value
        added is too large, it may disturb the PSI calculation.

    missing_values: str, default=ignore
        Whether to perform the PSI feature selection on a dataframe with missing values.
        Takes values 'raise' or 'ignore'. If 'ignore', missing values will be dropped
        when determining the PSI for that particular feature. If 'raise' the transformer
        will raise an error and features will not be selected.

    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        numerical variables in the dataset.

    Attributes
    ----------
    features_to_drop_:
        Set with the features that will be dropped.

    variables_:
        The variables to consider for the feature selection.

    psi_values_:
        Dictionary containing the PSI values for all features considered.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find features with high PSI values.
    transform:
        Remove features with high PSI values.
    fit_transform:
        Fit to the data. Then transform it.

    See Also
    --------
    To know more about the discretization visit:

    feature_engine.discretisation.EqualFrequencyDiscretiser
    feature_engine.discretisation.EqualWidthDiscretiser
    """

    def __init__(
        self,
        split_col: str = None,
        split_frac: float = 0.5,
        split_distinct: bool = False,
        cut_off: Union[None, int, float, datetime.date, List] = None,
        switch: bool = False,
        threshold: float = 0.25,
        bins: int = 10,
        strategy: str = "equal_frequency",
        min_pct_empty_buckets: float = 0.0001,
        missing_values: str = "ignore",
        variables: Variables = None,
    ):

        if split_col and not isinstance(split_col, (str, int)):
            raise ValueError(
                f"split_col must be a string an integer or None. Got "
                f"{split_col} instead."
            )

        # split_frac and cut_off can't be None at the same time
        if not split_frac and not cut_off:
            raise ValueError(
                "cut_off and split_frac cannot be both set to None. Please specify a "
                "value for at least one of these parameters."
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
                f"split_distinct_value must be a boolean. Got {split_distinct} "
                f"instead."
            )

        if not isinstance(switch, bool):
            raise ValueError(f"switch must be a boolean. Got {switch} instead.")

        if not isinstance(threshold, (float, int)) or threshold < 0:
            raise ValueError(
                f"threshold must be >= 0. Got {threshold} instead."
            )

        if not isinstance(bins, int) or bins <= 1:
            raise ValueError(f"bins must be an integer >= 1. Got {bins} "
                             f"instead.")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "strategy takes only values equal_width or equal_frequency. Got "
                f"{strategy} instead."
            )

        if (
            not isinstance(min_pct_empty_buckets, (float, int))
            or min_pct_empty_buckets < 0
        ):
            raise ValueError(
                f"min_pct_empty_buckets must be >= 0. Got {min_pct_empty_buckets} "
                f"instead."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                f"missing_values takes only values 'raise' or 'ignore'. Got "
                f"{missing_values} instead."
            )

        # Check the variables before assignment.
        self.variables = _check_input_parameter_variables(variables)

        # Set all remaining arguments as attributes.
        self.split_col = split_col
        self.split_frac = split_frac
        self.split_distinct_value = split_distinct
        self.cut_off = cut_off
        self.switch = switch
        self.threshold = threshold
        self.bins = bins
        self.strategy = strategy
        self.min_pct_empty_buckets = min_pct_empty_buckets
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Assess the features that needs to be dropped because of high PSI values.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : pandas series. Default = None
            y is not needed in this transformer. You can pass y or None.

        Returns
        -------
        self
        """
        # check input dataframe
        X = _is_dataframe(X)
        self.n_features_in_ = X.shape[1]

        # find all variables or check those entered are present in the dataframe
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        # Remove the split columns from the variables list. It is automatically added to
        # it if variables is not defined at initiation.
        if self.split_col in self.variables_:
            self.variables_.remove(self.split_col)

        if self.missing_values == "raise":
            # check if dataset contains na or inf
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        # Split the dataframe into a basis and a measurement dataframe.
        basis_df, measurement_df = self._split_dataframe(X)

        # Switch base and measurement dataframe if required.
        if self.switch:
            measurement_df, basis_df = basis_df, measurement_df

        # Compute the PSI by looping over the features
        self.psi_values_ = {}
        self.features_to_drop_ = []
        for feature in self.variables_:
            # Discretize the features using bucketing.
            basis_discrete, meas_discrete = self._discretize_features(
                basis_df[[feature]], measurement_df[[feature]]
            )
            # Approximate the features' normalized distribution.
            meas_distrib, basis_distrib = self._approximate_features_distribution(
                basis_discrete, meas_discrete
            )
            # Calculate the PSI value based on the distributions
            self.psi_values_[feature] = np.sum(
                (meas_distrib - basis_distrib) * np.log(meas_distrib / basis_distrib)
            )
            # Assess if the feature needs to be dropped
            if self.psi_values_[feature] > self.threshold:
                self.features_to_drop_.append(feature)

        return self

    def _discretize_features(self, series_basis, series_meas):
        """
        Use binning to discretise the values of the feature to invesigate.

        Parameters
        ----------
        series_basis : pandas series
            Series that will serve as basis for the PSI calculations (i.e. reference).

        series_meas : pandas series
            Series that will compared to the reference during PSI calculations.

        Returns
        -------
        basis_discrete, pd.Series.
            Series with discretised values for series_basis.
        meas_discrete, pd.Series.
            Series with discretised values for series_meas.
        """
        if self.strategy in ["equal_width"]:
            bucketer = EqualWidthDiscretiser(bins=self.bins)
        else:
            bucketer = EqualFrequencyDiscretiser(q=self.bins)

        # Discretize the features.
        basis_discrete = bucketer.fit_transform(series_basis.dropna())
        meas_discrete = bucketer.transform(series_meas.dropna())

        return basis_discrete, meas_discrete

    def _approximate_features_distribution(self, basis, meas):
        """
        Approximate the distribution of the two features.

        Parameters
        ----------
        basis : pd.Series.
            Series with discretised (i.e. binned) values.

        meas: pd.Series.
            Series with discretised (i.e. binned) values.

        Returns
        -------
        distribution.basis: pd.Series.
            Normalized distribution of the basis Series over the bins.

        distribution.meas: pd.Series.
            Normalized distribution of the meas Series over the bins.
        """
        # TODO: were the fillna in previous version needed?
        basis_distrib = basis.value_counts(normalize=True)
        meas_distrib = meas.value_counts(normalize=True)

        # Align the two distributions by merging the buckets (bins). This ensures
        # the number of bins is the same for the two distributions (in case of
        # empty bucket).
        distributions = (
            pd.DataFrame(basis_distrib)
            .merge(
                pd.DataFrame(meas_distrib),
                right_index=True,
                left_index=True,
                how="outer",
            )
            .fillna(self.min_pct_empty_buckets)
        )
        distributions.columns = ["basis", "meas"]

        return distributions.basis, distributions.meas

    def _split_dataframe(self, X):
        """
        Split a dataframe according to a cut-off value and return two dataframes:
        one with values above the cut-off and one with values below or equal
        to the cut-off.
        The cut-off value is associated to a specific column.

        The cut-off can be defined in two ways:

            - Considering all observations. In that case a split fraction of 0.25
            will ensure 25% of the observations are in the "below cut-off"
            dataframe. The final number of observations in the "below cut-off" may
            be slightly different than 25% as all observations with the same value
            are put in the same split.

            - Considering the values of the observations. The cut-off applies to
            the distinct values. In that case a split fraction of 0.25 will ensure
            25% of the distinct values are in the "below cut-off" dataframe;
            regardless of the number of observations.

        Parameters
        ----------
        X : pandas dataframe
            The original dataset. Correspond either to the measurement dataset
            or will be spit into a measurement and a basis dataframe.

        Returns
        -------
        within_cut_off, pd.DataFrame
            pandas dataframe with value within the cut_off
        outside_cut_off, pd.DataFrame
            pandas dataframe with value outside the cut_off
        """
        # Identify the values according to which the split must be done.
        if not self.split_col:
            reference = pd.Series(X.index)
        else:
            reference = X[self.split_col]

        # Raise an error if there are missing values in the reference column.
        if reference.isna().sum() != 0:
            raise ValueError(
                "Na's are not allowed in the columns used to split the dataframe."
            )

        # If no cut_off is pre-defined, compute it.
        if not self.cut_off:
            self.cut_off = self._get_cut_off_value(reference)

        # Split the original dataframe in two parts: within and outside cut-off
        if isinstance(self.cut_off, list):
            is_within_cut_off = reference.isin(self.cut_off)
        else:
            is_within_cut_off = reference <= self.cut_off

        within_cut_off = X[is_within_cut_off]
        outside_cut_off = X[~is_within_cut_off]

        return within_cut_off, outside_cut_off

    def _get_cut_off_value(self, ref):
        """
        Define the cut-off of the series (ref) in order to split the dataframe.

        - For a float or integer, np.quantile is used.
        - For other type, the quantile is based on the value_counts method.
        This allows to deal with date or string in a unified way.
            - The distinct values are sorted and the cumulative sum is
            used to compute the quantile. The value with the quantile that
            is the closest to the chosen split fraction is used as cut-off.
            - The sort involves that categorical values are sorted alphabetically
            and cut accordingly.

        Parameters
        ----------
        ref : (np.array, pd.Series).
            Series for which the nth quantile must be computed.

        Returns
        -------
        cut_off: (float, int, str, object).
            value for the cut-off.
        """
        # In case split_distinct_value is used, extract series with unique values
        if self.split_distinct_value:
            ref = pd.Series(ref.unique())

        # If the value is numerical, use numpy functionalities
        if isinstance(ref.iloc[0], (int, float)):
            cut_off = np.quantile(ref, self.split_frac)

        # Otherwise use value_counts combined with cumsum
        else:
            reference = pd.DataFrame(
                ref.value_counts(normalize=True).sort_index().cumsum()
            )

            # Get the index (i.e. value) with the quantile that is the closest
            # to the split_frac defined at initialization.
            distance = abs(reference - self.split_frac)
            cut_off = (distance.idxmin()).values[0]

        return cut_off
