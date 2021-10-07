from typing import List, Union

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
    """
    DropHighPSIFeatures drops features with a Population Stability Index (PSI) above
    a given threshold. The PSI of a feature is an indication of the shift in its
    distribution; a feature with high PSI might therefore be seen as unstable.

    In fields like Credit Risk Modelling, the elimination of features with high PSI
    is frequent and usually required by the Regulator.

    The calculation of the PSI requires to compare two distributions.
    In DropHighPSIFeatures the approach implemented is the following:

    - The dataframe is split in two parts and the PSI calculations are performed by
    comparing the distribution of the features in the two parts.

    - The user defines the columns of the dataframe according to which the dataframe
    will de split. The column can contain numbers, dates and strings.

    - The user defines a split fraction. This is used to define the sizes of the two
    dataframe. Is the split fraction is 50%, the two dataframe will have about the same
    size. It is not guaranteed that the size will exactly match because the observations
    are grouped by their label (i.e. their values in the split columns) so all
    observations with the same label will automatically belong to the same dataframe.

    - If the option split_distinct_value is activated, the number of distinct values
    is used to make the split. For example a feature with values: [1, 2, 3, 4, 4, 4]
    a 50% split without split_distinct_value will split into two parts [1, 2, 3] and
    [4, 4, 4] while if the option is set on, the split will be [1, 2] and [3, 4, 4, 4].

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

    psi_:
        Dataframe containing the PSI values for all features considered.

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
        variables: Variables = None,
        missing_values: str = "ignore",
        switch: bool = False,
        threshold: float = 0.25,
        bins: int = 10,
        strategy: str = "equal_frequency",
        min_pct_empty_buckets: float = 0.0001,
    ):
        self._check_init_values(
            split_col,
            split_frac,
            split_distinct_value,
            variables,
            missing_values,
            switch,
            threshold,
            bins,
            strategy,
            min_pct_empty_buckets,
        )
        # Set all arguments (except self and variables as attributes)
        for name, value in vars().items():
            if name not in ("self", "variables"):
                setattr(self, name, value)

        # Check the input is in the correct format.
        self.variables = _check_input_parameter_variables(variables)

        # If variable is not defined, the split_col will be part of the list.
        # It then needs to be removed because the selector will not act on it.
        self.variables = [var for var in self.variables if var != split_col]

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

        # find all variables or check those entered are present in the dataframe
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        # Split the dataframe into a basis and a measurement dataframe.
        basis_df, measurement_df = self._split_dataframe(X)

        # Switch base and measurement dataframe if required.
        if self.switch:
            measurement_df, basis_df = basis_df, measurement_df

        # Compute the PSI
        self.psi_ = self._compute_PSI(basis_df, measurement_df, self.bucketer)

        # Select features below the threshold
        self.features_to_drop_ = self.psi_[
            self.psi.value >= self.threshold
        ].index.to_list()

        self.n_features_in_ = X.shape[1]

        return self

    def _compute_PSI(self, df_basis, df_meas, bucketer):
        """
        Compute the PSI by first bucketing (=binning) and then call the psi calculator
        for each feature.

        Parameters
        ----------
        df_basis : pandas dataframe of shape = [m_rows, n_features]
            Serve as reference for the features distribution.

        df_meas : pandas dataframe of shape = [p_rows, n_features]
            Matrix for which the feature distributions will be compared to the basis.

        bucketer : EqualFrequencyDiscretiser or EqualWidthDiscretiser.
                Default = EqualFrequencyDiscretiser
            Class used to bucket (bin) the features in order to approximate
            the distribution.

        Returns
        -------
        results_df: pandas dataframe.
            Dataframe containing the PSI for each feature.
        """
        # Initialize a container for the results.
        results = {}

        # Compute the PSI for each feature excluding the column used for split.
        for feature in self.variables_:
            results[feature] = [
                self._compute_feature_psi(df_basis[[feature]], df_meas[[feature]])
            ]

        # Transform the result container in a user friendly format.
        results_df = pd.DataFrame.from_dict(results).T
        results_df.columns = ["value"]

        return results_df

    def _compute_feature_psi(self, series_basis, series_meas):
        """
        Call the PSI calculator for two distributions

        Parameters
        ----------
        series_basis : pandas series
            Proxy for the reference distribution.

        series_meas : pandas series
            Proxy for the measurement distribution.

        Returns
        -------
        psi_value: float.
            PSI value.
        """
        # Perform the binning for all features.
        basis_binned = (
            self.bucketer.fit_transform(series_basis.dropna()).value_counts().fillna(0)
        )

        meas_binned = (
            self.bucketer.transform(series_meas.dropna()).value_counts().fillna(0)
        )

        # Combine the two distributions by merging the buckets (bins)
        binning = (
            pd.DataFrame(basis_binned)
            .merge(
                pd.DataFrame(meas_binned),
                right_index=True,
                left_index=True,
                how="outer",
            )
            .fillna(0)
        )
        binning.columns = ["basis", "meas"]

        # Compute the PSI value.
        psi_value = self._psi(binning.basis.to_numpy(), binning.meas.to_numpy())

        return psi_value

    def _psi(self, d_basis, d_meas):
        """
        Compute the PSI value

        Parameters
        ----------
        d_basis : np.array.
            Proxy for the basis distribution.

        d_meas: np.array.
            Proxy for the measurement distribution.

        Returns
        -------
        psi_value: float.
            PSI value.
        """
        # Calculate the ratio of samples in each bin
        basis_ratio = d_basis / d_basis.sum()
        meas_ratio = d_meas / d_meas.sum()

        # Necessary to avoid divide by zero and ln(0). Has minor impact on PSI value.
        basis_ratio = np.where(
            basis_ratio <= 0, self.min_pct_empty_buckets, basis_ratio
        )
        meas_ratio = np.where(meas_ratio <= 0, self.min_pct_empty_buckets, meas_ratio)

        # Calculate the PSI value
        psi_value = np.sum(
            (meas_ratio - basis_ratio) * np.log(meas_ratio / basis_ratio)
        )

        return psi_value

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
        pandas dataframe with value below the
        basis pandas dataframe
        """
        # Identify the values according to which the split must be done.
        if self.split_col is None:
            reference = pd.Series(X.index.to_list()).dropna()
        else:
            reference = X[self.split_col].dropna()

        # Define the cut-off point based on quantile.
        if self.split_distinct_value:
            cut_off = self._get_cut_off_value(reference.unique())
        else:
            cut_off = self._get_cut_off_value(reference)

        self.quantile = {self.split_col: cut_off}

        # Split the original dataframe in two parts: above and below cut-off
        is_above_cut_off = reference > cut_off

        below_cut_off = X[~is_above_cut_off]
        above_cut_off = X[is_above_cut_off]

        return below_cut_off, above_cut_off

    def _get_cut_off_value(self, ref):
        """
        Define the cut-off of the series (ref) in order to split the dataframe.

        - For a float or integer, np.quantile is used.
        - For other type, the quantile is based on the value_counts method.
        This allows to deal with date or string in a unified way.
            - The distinct values are sorted and the cumulative sum is
            used to compute the quantile. The value with the quantile that
            is the closest to the chosen split fraction is used as cut-off.
            - The sort involves that categorical values are sorted alphabetically.

        Parameters
        ----------
        ref : (np.array, pd.Series).
            Series for which the nth quantile must be computed.

        Returns
        -------
        cut_off: (float, int, str, object).
            value for the cut-off.
        """
        # Ensure the argument is a pd.Series
        if not isinstance(ref, pd.Series):
            ref = pd.Series(ref)

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

    def _check_init_values(
        self,
        split_col,
        split_frac,
        split_distinct_value,
        variables,
        missing_values,
        switch,
        threshold,
        bins,
        strategy,
        min_pct_empty_buckets,
    ):
        """
        Raise an error if one of the arguments is not of the expected type or
        has an inadequate value.

        split_col: string.
        Label of the column according to which the dataframe will be split.

        split_frac: float.
            Ratio of the observations (when split_distinct is not activated) that
            goes into the sub-dataframe that is used to determine the reference for
            the feature distributions. The second sub-dataframe will be used to compare
            its feature distributions to the reference ones.

        split_distinct_values: boolean.
            If set on, the split fraction does not account for the number of
            observations but only for the number of distinct values in split_col.

        variables: list
            The list of variables to evaluate. If None, the transformer will
            evaluate all numerical variables in the dataset.

        switch: boolean
            If set to true the role of the two matrices involved in the PSI
            calculations (basis and measurement) will be switch. This is an
            important option as the PSI value is not symmetric
            (i.e. PSI(a, b) != PSI(b, a)).

        threshold: float
            Threshold above which the distribution of a feature has changed so
            much that the feature will be dropped. The most common values are
            0.25 (large shift) and 0.10 (medium shift).

        strategy: string or callable
            Type of binning used to represent the distribution of the feature.
            In can be either "equal_width" for equally spaced bins or
            "equal_frequency" for bins based on quantiles.

        bins: int
            Number of bins used in the binning. For numerical feature a value of
            10 is considered as appropriate. For features with lower cardinality
            lower values are usually used.

        min_pct_empty_buckets: float
            Value to add to empty bucket (when considering percentages). If a bin
            is empty the PSI value may jump to infinity. By adding a small number
            to empty bins, this issue is avoided. If the value added is too large,
            it may disturb the calculations.

        missing_values: str
            Takes values 'raise' and 'ignore'. Whether the missing values should
            be raised as error or ignored when determining correlation.

        Returns:
            None

        """
        if not isinstance(bins, int) or bins <= 1:
            raise ValueError("bins must be an integer larger than 1.")

        if not isinstance(switch, bool):
            raise ValueError("The value of switch basis must be True or False.")

        if not isinstance(threshold, (float, int)) or threshold < 0:
            raise ValueError("threshold must be a float larger than 0")

        if not isinstance(split_col, str):
            raise ValueError("split_col must be a string")

        if not 0 < split_frac < 1:
            raise ValueError("split_frac must be larger than 0 and smaller than 1")

        if not isinstance(split_distinct_value, bool):
            raise ValueError("split_distinct_value must be a boolean")

        if (
            not isinstance(min_pct_empty_buckets, (float, int))
            or min_pct_empty_buckets < 0
        ):
            raise ValueError("min_pct_empty_buckets must be larger or equal to 0")

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values can only be 'raise' or 'ignore'.")
        if strategy.lower() in ["equal_width"]:
            self.bucketer = EqualWidthDiscretiser(bins=bins)
        elif strategy.lower() in ["equal_frequency"]:
            self.bucketer = EqualFrequencyDiscretiser(q=bins)
        else:
            raise ValueError("Strategy must be either equal_width or equal_frequency")
