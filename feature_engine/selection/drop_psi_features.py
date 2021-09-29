from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na, 
    _is_dataframe,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser, EqualWidthDiscretiser
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
    distribution; a feature with high PSI might therefore be seen as instable.

    In fields like Credit Risk Modelling, the elimination of features with high PSI
    is frequent and usually required by the Regulator.

    The calculation of the PSI requires to compare two distributions.
    In DropHighPSIFeatures two approaches are implemented though the
    "basis" argument.

    - If it is a pandas.DataFrame, the class will compare the distributions of the
    X dataframe (argument of the fit method) and "basis". The two dataframes must
    contain the same features (i.e. labels).

    - If it is a dictionary, the X matrix of the fit method is split in two according
    to time and the distribution between the two parts are compared using the PSI.
    The dictionary must contain the label of a column with dates and the cut-off date.

    - The PSI calculations are not symmetric. The switch_basis argument allows to
    switch the role of the two dataframes in the PSI calculations.

    The comparison of the distribution is done through binning. Two strategies are
    implemented: equal_frequency and equal_width. These labels refer to two
    discretisation implementation from the present package.

    References:
    https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations

    DropHighPSIFeatures() works only with numerical variables. Categorical variables
    will need to be encoded to numerical or will be excluded from the analysis.

    Parameters
    ----------

    basis: pd.DataFrame or dictionary.
        Information required to define the basis for the PSI calculations. It is
        either a dataframe in the case of direct comparison or the label of the
        column containing the date and the cut-off-date in case a single
        dataframe is provided. In the latter case, the dataframe will be split
        on two parts that are non-overlapping over time.

    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        numerical variables in the dataset.

    switch_basis: boolean, default=False.
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
        Takes values 'raise' and 'ignore'. Whether the missing values should be raised
        as error or ignored when determining correlation.

    Attributes
    ----------
    features_to_drop_:
        Set with the correlated features that will be dropped.

    correlated_feature_sets_:
        Groups of correlated features. Each list is a group of correlated features.

    variables_:
        The variables to consider for the feature selection.

    n_features_in_:
        The number of features in the train set used in fit.

    psi:
        Dataframe containing the PSI values for all features considered.

    Methods
    -------
    fit:
        Find features with high PSI values.
    transform:
        Remove features wth high PSI values.
    fit_transform:
        Fit to the data. Then transform it.
    """

    # TODO: Implement the check on the types of the cut-off dates and the date
    # column that need ot be the same.
    def __init__(
        self,
        basis,
        variables: Variables = None,
        missing_values: str = "include",
        switch_basis=False,
        threshold: float = 0.25,
        bins=10,
        strategy="equal_frequency",
        min_pct_empty_buckets=0.0001,
    ):
        self._check_init_values(
            basis, variables, missing_values, switch_basis, threshold, bins, strategy, min_pct_empty_buckets,
        )
        # Set all arguments (except self and variables as attributes)
        for name, value in vars().items():
            if name not in ("self", "variables"):
                setattr(self, name, value)

        # Check the input is in the correct format.
        self.variables = _check_input_parameter_variables(variables)

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

        # TODO: Test missing values do not lead to an error and have no impact.

        # Split the dataframe into a basis and a measurement dataframe.
        measurement_df, basis_df = self._split_dataframe(X, self.basis)

        # Switch base and measurement dataframe if required.
        if self.switch_basis:
            measurement_df, basis_df = basis_df, measurement_df

        # Compute the PSI
        self.psi = self._compute_PSI(basis_df, measurement_df, self.bucketer)

        # Select features below the threshold
        self.features_to_drop_ = self.psi[self.psi.value >= self.threshold].index.to_list()

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

        # Compute the PSI for each feature.
        for feature in self.variables_:
            results[feature] = [self._compute_feature_psi(df_basis[[feature]], df_meas[[feature]])]

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
        basis_binned = self.bucketer.fit_transform(series_basis.dropna()).value_counts().fillna(0)

        meas_binned = self.bucketer.transform(series_meas.dropna()).value_counts().fillna(0)

        # Combine the two distributions by merging the buckets (bins)
        binning = (
            pd.DataFrame(basis_binned)
            .merge(pd.DataFrame(meas_binned), right_index=True, left_index=True, how="outer",)
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
        basis_ratio = np.where(basis_ratio <= 0, self.min_pct_empty_buckets, basis_ratio)
        meas_ratio = np.where(meas_ratio <= 0, self.min_pct_empty_buckets, meas_ratio)

        # Calculate the PSI value
        psi_value = np.sum((meas_ratio - basis_ratio) * np.log(meas_ratio / basis_ratio))

        return psi_value

    def _split_dataframe(self, X, basis):
        """
        Split a dataframe according to a cut-off date in case only a single
        dataframe is provided. Otherwise return the two original dataframes.

        Parameters
        ----------
        X : pandas dataframe
            The original dataset. Correspond either to the measurement dataset
            or will be spit into a measurement and a basis dataframe.
        basis : pandas dataframe or dict
            If it is a dict, it must contain a reference to a date column and
            the cut-off date.

        Returns
        -------
        measurement pandas dataframe
        basis pandas dataframe
        """
        # In case a dataframe is passed do nothing.
        if isinstance(basis, pd.DataFrame):
            return X, basis

        # In case a dictionary is passed, split the dataframe according to time.
        elif isinstance(basis, dict):
            date_col = basis["date_col"]
            value = basis["cut_off_date"]

            below_value = X[X[date_col] <= value]
            above_value = X[X[date_col] > value]

            return above_value, below_value

        # In any other cases raise an error.
        else:
            raise ValueError("compare should be either a pd.dataframe or a dictionary")

    def _check_init_values(
        self, basis, variables, missing_values, switch_basis, threshold, bins, strategy, min_pct_empty_buckets,
    ):
        """
        Raise an error if one of the arguments is not of the expected type or 
        has an inadequate value.

        basis: pd.DataFrame or dictionary.
            Information required to define the basis for the PSI calculations.
            It is either a dataframe in the case of direct comparison or the
            label of the column containing the date and the cut-off-date in
            case a single dataframe is provided. In the latter case,the
            dataframe will be split on two parts that arenon-overlapping
            over time.

        variables: list
            The list of variables to evaluate. If None, the transformer will
            evaluate all numerical variables in the dataset.

        switch_basis: boolean
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

        if not isinstance(switch_basis, bool):
            raise ValueError("The value of switch basis must be True or False.")

        if not isinstance(threshold, (float, int)) or threshold < 0:
            raise ValueError("threshold must be a float larger than 0")

        if not isinstance(basis, (pd.DataFrame, dict)):
            raise ValueError("basis must be either a pd.DataFrame or a dict")

        if isinstance(basis, dict):
            if "date_col" not in basis.keys() or "cut_off_date" not in basis.keys():
                raise ValueError("date_col and cut_off_date must be keys")

        if not isinstance(min_pct_empty_buckets, (float, int)) or min_pct_empty_buckets < 0:
            raise ValueError("min_pct_empty_buckets must be larger or equal to 0")

        if missing_values not in ["raise", "ignore", "include"]:
            raise ValueError("missing_values can only be 'raise', 'ignore' or 'include'.")
        if strategy.lower() in ["equal_width"]:
            self.bucketer = EqualWidthDiscretiser(bins=bins)
        elif strategy.lower() in ["equal_frequency"]:
            self.bucketer = EqualFrequencyDiscretiser(q=bins)
        else:
            raise ValueError("Strategy must be either equal_width or equal_frequency")
