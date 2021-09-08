from typing import List, Union

import pandas as pd

from feature_engine.dataframe_checks import _check_contains_na, _is_dataframe
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


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
        date_column: string = None,
        date_cut_off: object = None,
        threshold: float = 0.2, 
        bucketer: object = LinearBucketer(n_bins=10),
        min_value: float = 0.0001,
        missing_values: str = "raise"
    ):

        if not isinstance(threshold, (float, int)) or threshold < 0:
            raise ValueError("threshold must be a float larger than 0")

        if missing_values not in ["raise", "ignore", "include"]:
            raise ValueError(
                "missing_values takes only values 'raise', 'ignore' or " "'include'."
            )
        self.min_value = min_value
        self.date_column = date_column
        self.date_cut_off = date_cut_off
        self.bucketer = bucketer
        self.threshold = threshold
        self.variables = _check_input_parameter_variables(variables)
        self.missing_values = missing_values

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
        self.variables_ = _find_all_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)

        if self.missing_values == "include":
            X[self.variables_] = X[self.variables_].fillna("missing_values")

        # Split the dataframe into a reference and a comparison
        reference_df, comparison_df = self._split_dataframe(X, self.date_column, self.date_cut_off)
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
            results{feature: self._compute_feature_psi(df_ref[feature], df_comp[feature], bucketer)}

        results_df = pd.DataFrame(results).T
        results_df.columns = ['value']

        return results_df


    def _compute_feature_psi(self, series_ref, series_comp, bucketer):
        bucketer.fit(series_ref)
        bincounts_ref = bucketer.fit_compute(series_ref)
        bincounts_comp = bucketer.compute(series_comp)

        psi_value = self._psi(bincounts_ref, bincounts_comp)

        return psi_value

    def _psi(self, d1, d2):
        # Calculate the ratio of samples in each bin
        ref_ratio = d1 / d1.sum()
        comp_ratio = d2 / d2.sum()

        # Necessary to avoid divide by zero and ln(0). Should have minor impact on PSI value.
        ref_ratio = np.where(ref_ratio <=0, self.min_value, ref_ratio)
        comp_ratio = np.where(comp_ratio <=0, self.min_value, comp_ratio)

        # Calculate the PSI value
        psi_value = np.sum((comp_ratio - ref_ratio) * np.log(comp_ratio / ref_ratio))

        return psi_value

    def _split_dataframe(self, X, column, value):

        below_value = X[X[column] <= value]
        above_value = X[X[column] > value]

        return below_value, above_value


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


import pandas as pd
import numpy as np

class LinearBucketer():
    """
    Create equally spaced bins using numpy.histogram function.
    Example:
    ```python
    from probatus.binning import SimpleBucketer
    x = [1, 2, 1]
    bins = 3
    myBucketer = SimpleBucketer(bin_count=bins)
    myBucketer.fit(x)
    ```
    myBucketer.counts gives the number of elements per bucket
    myBucketer.boundaries gives the boundaries of the buckets
    """

    def __init__(self, bin_count):
        """
        Init.
        """
        self.bin_count = bin_count


    def _linear_bins(self, x, bin_count):
        """
        Simple bins.
        """
        _, boundaries = np.histogram(x, bins=bin_count)
        # Ensure all cases are included in the lower and upper bound
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        counts = self._compute_counts_per_bin(x, boundaries)
        return counts, boundaries

    def _compute_counts_per_bin(X, boundaries):
        """
        Computes the counts per bin.
        Args:
            X (np.array): data to be bucketed
            boundaries (np.array): boundaries of the bins.
        Returns (np.array): Counts per bin.
        """
        # np.digitize returns the indices of the bins to which each value in input array belongs
        # the smallest value of the `boundaries` attribute equals the lowest value in the set the instance was
        # fitted on, to prevent the smallest value of x_new to be in his own bucket, we ignore the first boundary
        # value
        bins = len(boundaries) - 1
        digitize_result = np.digitize(X, boundaries[1:], right=True)
        result = pd.DataFrame({"bucket": digitize_result}).groupby("bucket")["bucket"].count()
        # reindex the dataframe such that also empty buckets are included in the result
        return result.reindex(np.arange(bins), fill_value=0).to_numpy()

    def fit(self, x, y=None):
        """
        Fit bucketing on x.
        Args:
            x: (np.array) Input array on which the boundaries of bins are fitted
            y: (np.array) ignored. For sklearn-compatibility
        Returns: fitted bucketer object
        """
        self.counts_, self.boundaries_ = self._linear_bins(x, self.bin_count)
        return self

    def compute(self, X, y=None):
        """
        Applies fitted bucketing algorithm on input data and counts number of samples per bin.
        Args:
            X: (np.array) data to be bucketed
            y: (np.array) ignored, for sklearn compatibility
        Returns: counts of the elements in X using the bucketing that was obtained by fitting the Bucketer instance
        """
        return self._compute_counts_per_bin(X, self.boundaries_)

    def fit_compute(self, X, y=None):
        """
        Apply bucketing to new data and return number of samples per bin.
        Args:
            X: (np.array) data to be bucketed
            y: (np.array) One dimensional array, used if the target is needed for the bucketing. By default is set to
            None
        Returns: counts of the elements in x_new using the bucketing that was obtained by fitting the Bucketer instance
        """
        self.fit(X, y)
        return self.compute(X, y)

    