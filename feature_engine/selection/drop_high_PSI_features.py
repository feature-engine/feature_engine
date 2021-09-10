from typing import List, Union

import pandas as pd
import numpy as np

from feature_engine.dataframe_checks import _check_contains_na, _is_dataframe
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.discretisation.equal_frequency import EqualFrequencyDiscretiser
from feature_engine.discretisation.equal_width import EqualWidthDiscretiser
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
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

    The calculation of the PSI requires to compare two distributions. In DropHighPSIFeatures
    two approaches are implemented though the "basis" argument.

    - If it is a pandas.DataFrame, the class will compare the distributions of the X dataframe
    (argument of the fit method) and "basis". The two dataframes must contain the same features
    (i.e. labels). 

    - If it is a dictionary, the X matrix of the fit method is split in two according to time
    and the distribution between the two parts are compared using the PSI. The dictionary must 
    contain the label of a column with dates and the cut-off date.

    - The PSI calculations are not symmetric. The switch_basis argument allows to switch the 
    role of the two dataframes in the PSI calculations.

    The comparison of the distribution is done through binning. Two strategies are implemented:
    equal_frequency and equal_width. These labels refer to two discretisation implementation from
    the present package. 

    References:
    - [Statistical Properties of Population Stability Index](https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations)

    """

    def __init__(self, basis, variables: Variables = None, missing_values: str = "raise",
    switch_basis=False, threshold: int = 0.25,n_bins = 10, method = 'equal_frequency',
    min_pct_empty_buckets = 0.0001):

        # Set all arguments (except self and variables as attributes)
        [setattr(self,name,value) for name,value in vars().items() if name not in ("self", 'variables')]
        
        self.variables = _check_input_parameter_variables(variables)

        self._check_init_values(basis, variables, missing_values, switch_basis, 
        threshold, n_bins, method, min_pct_empty_buckets)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are present in the dataframe
        self.variables_ =  _find_or_check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)

        if self.missing_values == "include":
            X[self.variables_] = X[self.variables_].fillna("missing_values")

        # Split the dataframe into a reference and a comparison if required.
        measurement_df, basis_df = self._split_dataframe(X, self.basis)

        # Switch base and measurement dataframe if required. Remind that PSI
        # if not symmetric so PSI(a, b) != PSI(b, a) except if a and b have the
        # same binning.
        if self.switch_basis:
            measurement_df, basis_df = basis_df, measurement_df
        # Compute the PSI
        self.psi = self._compute_PSI(basis_df, measurement_df, self.bucketer)
        
        # Select features below the threshold
        self.features_to_drop_ = self.psi[self.psi.value >= self.threshold].index.to_list()

        self.n_features_in_ = X.shape[1]

        return self

    def _compute_PSI(self,df_ref, df_comp, bucketer):

        ref = bucketer.fit_transform(df_ref).fillna(0)
        comp = bucketer.transform(df_comp).fillna(0)

        results = {}

        for feature in self.variables_:
            results[feature] = [self._compute_feature_psi(
                ref[[feature]].value_counts(), 
                comp[[feature]].value_counts(), bucketer)]

        results_df = pd.DataFrame.from_dict(results).T
        results_df.columns = ['value']

        return results_df


    def _compute_feature_psi(self, series_ref, series_comp, bucketer):

        binning = pd.DataFrame(series_ref).merge(pd.DataFrame(series_comp), 
        right_index=True, left_index=True, how="outer"
        ).fillna(0)
        binning.columns = ['ref', 'comp']

        psi_value = self._psi(binning.ref.to_numpy(), binning.comp.to_numpy())

        return psi_value

    def _psi(self, d1, d2):
        # Calculate the ratio of samples in each bin
        ref_ratio = d1 / d1.sum()
        comp_ratio = d2 / d2.sum()

        # Necessary to avoid divide by zero and ln(0). Should have minor impact on PSI value.
        ref_ratio = np.where(ref_ratio <=0, self.min_pct_empty_buckets, ref_ratio)
        comp_ratio = np.where(comp_ratio <=0, self.min_pct_empty_buckets, comp_ratio)

        # Calculate the PSI value
        psi_value = np.sum((comp_ratio - ref_ratio) * np.log(comp_ratio / ref_ratio))

        return psi_value

    def _split_dataframe(self, X, basis):

        if isinstance(basis, pd.DataFrame):
            return X, basis
        elif isinstance(basis, dict):
            date_col = basis["date_col"]
            value = basis['cut_off_date']

            below_value = X[X[date_col] <= value]
            above_value = X[X[date_col] > value]
            
            return above_value, below_value

        else:
            raise ValueError("compare should be either a pd.dataframe or a dictionary")

    def _check_init_values(self, basis, variables, missing_values, switch_basis, threshold, 
    n_bins, method, min_pct_empty_buckets):

        if not isinstance(n_bins, int) or n_bins <= 1:
            raise ValueError("n_bins must be an integer larger than 1.")

        if not isinstance(switch_basis, bool):
            raise ValueError("The value of switch basis must be True or False.")

        if not isinstance(threshold, (float, int)) or threshold < 0:
            raise ValueError("threshold must be a float larger than 0")

        if not isinstance(basis, (pd.DataFrame, dict)):
            raise ValueError("basis must be either a pd.DataFrame or a dict")

        if isinstance(basis, dict):
            if 'date_col' not in basis.keys() or 'cut_off_date' not in basis.keys():
                raise ValueError("date_col and cut_off_date must be keys of the basis dictionary")

        if not isinstance(min_pct_empty_buckets, (float, int)) or min_pct_empty_buckets < 0:
            raise ValueError("min_pct_empty_buckets must be larger or equal to 0")

        if missing_values not in ["raise", "ignore", "include"]:
            raise ValueError(
                "missing_values takes only values 'raise', 'ignore' or " "'include'."
            )
        if method.lower() in ["equalwidth", 'equal_width', "equal width"]:
            self.bucketer = EqualWidthDiscretiser(bins=n_bins)
        elif method.lower() in  ["equalfrequency", 'equal_frequency', "equal frequency"]:
            self.bucketer = EqualFrequencyDiscretiser(q=n_bins)
        else:
            raise ValueError("Incorrect name for the method, should be either equal_width or equal_frequency")