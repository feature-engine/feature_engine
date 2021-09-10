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

class dvl():

    def __init__(self, basis, variables: Variables = None, missing_values: str = "raise",
    switch_basis=False, threshold: int = 0.25,n_bins = 10, method = 'equal_frequency'):

        if not isinstance(threshold, (float, int)) or threshold < 0:
            raise ValueError("threshold must be a float larger than 0")

        if missing_values not in ["raise", "ignore", "include"]:
            raise ValueError(
                "missing_values takes only values 'raise', 'ignore' or " "'include'."
            )
        
        self._min_value = 0.0001
        self._basis = basis
        self.switch_basis = switch_basis
        self.threshold = threshold
        self.variables = _check_input_parameter_variables(variables)
        self.missing_values = missing_values

        if method.lower() in ["equalwidth", 'equal_width', "equal width"]:
            self.bucketer = EqualWidthDiscretiser(bins=n_bins)
        elif method.lower() in  ["equalfrequency", 'equal_frequency', "equal frequency"]:
            self.bucketer = EqualFrequencyDiscretiser(q=n_bins)
        else:
            raise ValueError("Incorrect name for the method, should be either equal_width or equal_frequency")
        
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
        measurement_df, basis_df = self._split_dataframe(X, self._basis)

        # Switch base and measurement dataframe if required. Remind that PSI
        # if not symmetric so PSI(a, b) != PSI(b, a) except if a and b have the
        # same binning.
        if self.switch_basis:
            measurement_df, basis_df = basis_df, measurement_df
        # Compute the PSI
        self.psi = self._compute_PSI(basis_df, measurement_df, self.bucketer)
        
        # Select features below the threshold
        self.features_to_drop_ = self.psi[self.psi.value >= self.threshold].index.to_list()

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
        ref_ratio = np.where(ref_ratio <=0, self._min_value, ref_ratio)
        comp_ratio = np.where(comp_ratio <=0, self._min_value, comp_ratio)

        # Calculate the PSI value
        psi_value = np.sum((comp_ratio - ref_ratio) * np.log(comp_ratio / ref_ratio))

        return psi_value

    def _split_dataframe(self, X, basis):

        if isinstance(basis, pd.DataFrame):
            return X, basis
        elif isinstance(basis, dict):
            date_col = basis["date_col"]
            value = basis['cut_off']
            below_value = X[X[column] <= value]
            above_value = X[X[column] > value]
            
            return below_value, above_value

        else:
            raise ValueError ("compare should be either a pd.dataframe or a dictionary")

    def transform(self, X: pd.DataFrame):
        # check if input is a dataframe
        X = _is_dataframe(X)
        # return the dataframe with the selected features
        return X.drop(columns=self.features_to_drop_)

    def fit_transform(self, X):

        return self.fit(X).transform(X)



class DropHighPSIFeatures(BaseSelector):
    """
    

    Parameters
    ----------
    

    Attributes
    ----------
    features_to_drop_:
        List with constant and quasi-constant features.

    variables_:
        The variables to consider for the feature selection.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find constant and quasi-constant features.
    transform:
        Remove constant and quasi-constant features.
    fit_transform:
        Fit to the data. Then transform it.

    Notes
    -----
    This transformer is a similar concept to the VarianceThreshold from Scikit-learn,
    but it evaluates number of unique values instead of variance

    See Also
    --------
    sklearn.feature_selection.VarianceThreshold
    """

    def __init__(
        self, 
        variables: Variables = None, 
        missing_values: str = "raise",
        threshold: int = 0.25,
        n_bins = 10,
    ):

        assert n_bins == 10

        if not isinstance(threshold, (float, int)) or threshold < 0:
            raise ValueError("threshold must be a float larger than 0")

        if missing_values not in ["raise", "ignore", "include"]:
            raise ValueError(
                "missing_values takes only values 'raise', 'ignore' or " "'include'."
            )

        
        self._min_value = 0.0001
        self._compare = None
        self.threshold = threshold
        self.variables = _check_input_parameter_variables(variables)
        self.missing_values = missing_values

        method = 'equal_width'

        if method.lower() in ["equalwidth", 'equal_width', "equal width"]:
            self.bucketer = EqualWidthDiscretiser(10)
        elif method.lower() in  ["equalfrequency", 'equal_frequency', "equal frequency"]:
            self.bucketer = EqualFrequencyDiscretiser(10)
        else:
            raise ValueError("Incorrect name for the method, should be either equal_width or equal_frequency")

        return

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Find constant and quasi-constant features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.
        y: None
            y is not needed for this transformer. You can pass y or None.

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

        if self.missing_values == "include":
            X[self.variables_] = X[self.variables_].fillna("missing_values")

        # Split the dataframe into a reference and a comparison
        reference_df, comparison_df = self._split_dataframe(X, self._compare)
        # Compute the PSI
        self.psi = self.compute_PSI(reference_df, comparison_df, self.bucketer)
        
        # Select features below the threshold
        self.features_to_drop_ =self.psi[self.psi.value >= self.threshold].index.to_list()

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__

    def _compute_PSI(df_ref, df_comp, bucketer):

        results = {}
        
        for feature in self.variables:
            results['feature'] = self._compute_feature_psi(df_ref[feature], df_comp[feature], bucketer)

        results_df = pd.DataFrame(results).T
        results_df.columns = ['value']

        return results_df


    def _compute_feature_psi(self, series_ref, series_comp, bucketer):
        ref = bucketer.fit_transform(series_ref).value_counts()
        comp = bucketer.transform(series_comp)
        binning = ref.merge(comp).fillna(0)
        binning.columns = ['ref', 'comp']

        psi_value = self._psi(binning.ref.to_numpy(), binning.ref.to_numpy())

        return psi_value

    def _psi(self, d1, d2):
        # Calculate the ratio of samples in each bin
        ref_ratio = d1 / d1.sum()
        comp_ratio = d2 / d2.sum()

        # Necessary to avoid divide by zero and ln(0). Should have minor impact on PSI value.
        ref_ratio = np.where(ref_ratio <=0, self._min_value, ref_ratio)
        comp_ratio = np.where(comp_ratio <=0, self._min_value, comp_ratio)

        # Calculate the PSI value
        psi_value = np.sum((comp_ratio - ref_ratio) * np.log(comp_ratio / ref_ratio))

        return psi_value

    def _split_dataframe(self, X, compare):

        if isinstance(compare, pd.DataFrame):
            return X, compare
        elif isinstance(compare, dict):
            date_col = compare["date_col"]
            value = compare['cut_off']
            below_value = X[X[column] <= value]
            above_value = X[X[column] > value]
            
            return below_value, above_value

        else:
            raise ValueError ("compare should be either a pd.dataframe or a dictionary")

        


    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        tags_dict["_xfail_checks"][
            "check_fit2d_1feature"
        ] = "the transformer needs at least 2 columns to compare, ok to fail"
        tags_dict["_xfail_checks"][
            "check_fit2d_1sample"
        ] = "the transformer raises an error when dropping all columns, ok to fail"
        return tags_dict
    