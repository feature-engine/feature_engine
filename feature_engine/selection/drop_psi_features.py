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
    shift in its distribution; a feature with high PSI might therefore be seen as
    unstable. In order to compute the PSI we will split the feature in two parts:
    train and test. Train will be used to define the expected feature distribution
    while test will be used to assess the difference with the train distrbution.

    In fields like Credit Risk Modelling, the elimination of features with high PSI
    is common use and usually required by the Regulator.

    When comparing the test distribution of a feature to its train counterpart,
    the PSI is computed as follow:

    - Define bins and discretise the feature from the train set.
    - Apply the binning to the same feature from the test set.
    - Compute the percentage of the distribution in each bin (test_i
      and train_i for bin i).
    - Compute the PSI using the following formula:

    PSI = \sum_{i=1}^n (test_i - train_i) . ln(\frac{test_i}{train_i})

    Thresholds are used to assess the importance of the population shift reported by the
    PSI value. The most commonly used thresholds are:
    - Below 10%, the variable has not experienced a significant shift.
    - Above 25%, the variable has experienced a major shift.
    - Between those two values, the shift is intermediate.

    When working with PSI, the following 3 points are worth mentioning:
    - The PSI is not symmetric with respected to the actual and reference; flipping the
    order will lead to different values fir the PSI.
    - The number of bins has an impact on the PSI values. 10 bins is usually
    a good choice.
    - The PSI is suited for numerical feature (either continuous or with high
    cardinality). For categorical features we advise to compute the difference
    between distribution using Chi-Square or Kolmogorov-Smirnov.

    In DropHighPSIFeatures the approach implemented is the following:

    - The dataframe is split in two parts and the PSI calculations are performed by
    comparing the distribution of the features in the two parts.

    - The user defines the columns of the dataframe according to which the dataframe
    will de split. The column can contain numbers, dates and strings.

    - The user defines a split fraction or a cut-off value. This is used to define the
    sizes of the two dataframe.
        - If the split fraction is 50%, the two dataframe will have about the same
        size. It is not guaranteed that the size will exactly match because the
        observations are grouped by their label (i.e. their values in the split
        columns) so all observations with the same label will automatically belong
        to the same dataframe.
        - A cut-off value define the border between the train and the test set.
        It can be a value (float, int) and date or a list. In case of a list the
        element of the list define the train set.

    - If the option split_distinct_value is activated, the number of distinct values
    is used to make the split. For example a feature with values: [1, 2, 3, 4, 4, 4]
    a 50% split without split_distinct_value will split into two parts [1, 2, 3] and
    [4, 4, 4] while if the option is set on, the split will be [1, 2] and [3, 4, 4, 4].
    split_distinct is only meaningful in combination with split_fraction. It is
    irrelevant when a cut-off is used.

    The PSI calculations are not symmetric. The switch argument allows to
    switch the role of the two dataframes in the PSI calculations.

    The comparison of the distribution is done through binning. Two strategies are
    implemented: equal_frequency and equal_width. These labels refer to two
    discretisation implementation from the present package.

    References:
    https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations

    DropHighPSIFeatures suited for numerical variables and for discrete variables with
    high cardinality. For categorical variables differences in distribution can be
    investigated using a Chi-square or Kolmogorov-Smirnov test.

    Parameters
    ----------

    split_col: string, default=None.
        Label of the column according to which the dataframe will be split.

    split_frac: float, default=0.5.
        Ratio of the observations (when split_distinct is not activated) that goes
        into the sub-dataframe that is used to determine the reference for the feature
        distributions. The second sub-dataframe will be used to compare its feature
        distributions to the reference ones.

    split_distinct_values: boolean, default=False.
        If set on, the split fraction does not account for the number of observations
        but only for the number of distinct values in split_col.

    cut_off: None, int, float, date or list, default=None
        Threshold used to split the split_col values. If int, float or date the split
        is done by selecting the values below and starting from the threshold. If
        cut_off is a list, values belonging to the list will be split from the values
        not belonging to the list.
        cut_off is conflicting with split_frac as they both define a different way
        to split the dataframe for performing PSI calculations. So cases where the
        two arguments are None and cases where the two arguments are assigned to
        a value are not valid.

    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        numerical variables in the dataset.

    switch: boolean, default=False.
        If set to true the role of the two matrices involved in the PSI calculations
        (basis and measurement) will be switch. This is an important option as the
        PSI value is not symmetric (i.e. PSI(a, b) != PSI(b, a)).

    threshold: float, default = 0.25.
        Threshold above which the distribution of a feature has changed so much that
        the feature will be dropped. The most common values are 0.25 (large shift)
        and 0.10 (medium shift).

    strategy: string or callable, default='equal_frequency'
        Type of binning used to represent the distribution of the feature. In can be
        either "equal_width" for equally spaced bins or "equal_frequency" for bins
        based on quantiles.

    bins, int, default = 10
        Number of bins used in the binning. For numerical feature a value of 10 is
        considered as appropriate. For features with lower cardinality lower values
        are usually used.

    min_pct_empty_buckets, float, default = 0.0001
        Value to add to empty bucket (when considering percentages). If a bin is
        empty the PSI value may jump to infinity. By adding a small number to empty
        bins, this issue is avoided. If the value added is too large, it may disturb
        the calculations.

    missing_values: str, default=ignore
        Takes values 'raise' or 'ignore'. Whether the missing values should be raised
        as error or ignored when computing PSI.

    Attributes
    ----------
    features_to_drop_:
        Set with the correlated features that will be dropped.

    variables_:
        The variables to consider for the feature selection.

    n_features_in_:
        The number of features in the train set used in fit.

    psi_values_:
        Dictionary containing the PSI values for all features considered.

    Methods
    -------
    fit:
        Find features with high PSI values.
    transform:
        Remove features with high PSI values.
    fit_transform:
        Fit to the data. Then transform it.
    """

    def __init__(
        self,
        split_col: str = None,
        split_frac: float = 0.5,
        split_distinct_value: bool = False,
        cut_off: Union[None, int, float, datetime.date, List] = None,
        variables: Variables = None,
        missing_values: str = "ignore",
        switch: bool = False,
        threshold: float = 0.25,
        bins: int = 10,
        strategy: str = "equal_frequency",
        min_pct_empty_buckets: float = 0.0001,
    ):
        # Ensure the arguments are according to expectations
        if (cut_off and split_frac) and not (cut_off or split_frac):
            raise ValueError(
                "cut_off and split frac cannot be defined at the same time."
                f"The values provided for cut_off and split_frac are {cut_off} "
                f"and {split_frac}."
            )
        if not isinstance(bins, int) or bins <= 1:
            raise ValueError(f"bins must be than 1 but has value: {bins}.")

        if not isinstance(switch, bool):
            raise ValueError(f"Switch must be a boolean but has value: {switch}.")

        if not isinstance(threshold, (float, int)) or threshold < 0:
            raise ValueError(
                f"threshold must be larger than 0 but has value: {threshold}."
            )

        if not isinstance(split_col, (str, type(None))):
            raise ValueError(
                f"split_col must be a string but has type: {type(split_col)}"
            )

        if split_frac is not None:
            if not (0 < split_frac < 1):
                raise ValueError(
                    f"split_frac must be between 0 and 1 but is: {split_frac}"
                )

        if not isinstance(split_distinct_value, bool):
            raise ValueError(
                "split_distinct_value must be a boolean but is "
                f"{type(split_distinct_value)}"
            )

        if (
            not isinstance(min_pct_empty_buckets, (float, int))
            or min_pct_empty_buckets < 0
        ):
            raise ValueError(
                f"min_pct_empty_buckets must >= 0 but has value {min_pct_empty_buckets}"
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values can only be 'raise' or 'ignore' but is "
                f"{missing_values}."
            )

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "Strategy must be either equal_width or equal_frequency but "
                f"is {strategy}"
            )

        # Check the variables before assignment.
        self.variables = _check_input_parameter_variables(variables)

        # Set all remaining arguments as attributes.
        self.split_col = split_col
        self.split_frac = split_frac
        self.split_distinct_value = split_distinct_value
        self.cut_off = cut_off
        self.missing_values = missing_values
        self.switch = switch
        self.threshold = threshold
        self.bins = bins
        self.strategy = strategy
        self.min_pct_empty_buckets = min_pct_empty_buckets

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
