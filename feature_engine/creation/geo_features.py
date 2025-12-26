# Authors: Ankit Hemant Lade (contributor)
# License: BSD 3 clause

from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
)
from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import check_numerical_variables

# Earth's radius in different units
EARTH_RADIUS = {
    "km": 6371.0,
    "miles": 3958.8,
    "meters": 6371000.0,
    "feet": 20902231.0,
}


class GeoDistanceTransformer(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    GeoDistanceTransformer() calculates the distance between two geographical
    coordinate pairs (latitude/longitude) and adds the result as a new feature.

    This transformer is useful for location-based machine learning problems such as
    real estate pricing, delivery route optimization, ride-sharing applications,
    and any domain where geographic proximity is relevant.

    The transformer supports different distance calculation methods:

    - 'haversine': Great-circle distance using the Haversine formula (default).
      Most accurate for typical distances on Earth's surface.
    - 'euclidean': Simple Euclidean distance in the coordinate space.
      Fast but less accurate for long distances.
    - 'manhattan': Manhattan (taxicab) distance in coordinate space.
      Useful as a rough approximation for grid-based city layouts.

    More details in the :ref:`User Guide <geo_distance_transformer>`.

    Parameters
    ----------
    lat1: str
        Column name containing the latitude of the first point.

    lon1: str
        Column name containing the longitude of the first point.

    lat2: str
        Column name containing the latitude of the second point.

    lon2: str
        Column name containing the longitude of the second point.

    method: str, default='haversine'
        The distance calculation method. Options are:
        - 'haversine': Great-circle distance (most accurate)
        - 'euclidean': Euclidean distance in coordinate space
        - 'manhattan': Manhattan distance in coordinate space

    output_unit: str, default='km'
        The unit for the output distance. Options are:
        - 'km': Kilometers
        - 'miles': Miles
        - 'meters': Meters
        - 'feet': Feet

    output_col: str, default='geo_distance'
        Name of the new column containing the calculated distances.

    drop_original: bool, default=False
        Whether to drop the original coordinate columns after transformation.

    Attributes
    ----------
    variables_:
        List of the coordinate variables used for distance calculation.

    feature_names_in_:
        List with the names of features seen during fit.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        This transformer does not learn parameters. Validates input columns.

    fit_transform:
        Fit to data, then transform it.

    transform:
        Calculate distances and add them as a new column.

    get_feature_names_out:
        Get output feature names for transformation.

    See Also
    --------
    feature_engine.creation.MathFeatures :
        Combines existing features using mathematical operations.
    feature_engine.creation.RelativeFeatures :
        Creates features relative to reference variables.

    References
    ----------
    .. [1] Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.creation import GeoDistanceTransformer
    >>> X = pd.DataFrame({
    ...     'origin_lat': [40.7128, 34.0522, 41.8781],
    ...     'origin_lon': [-74.0060, -118.2437, -87.6298],
    ...     'dest_lat': [34.0522, 41.8781, 40.7128],
    ...     'dest_lon': [-118.2437, -87.6298, -74.0060],
    ... })
    >>> gdt = GeoDistanceTransformer(
    ...     lat1='origin_lat', lon1='origin_lon',
    ...     lat2='dest_lat', lon2='dest_lon',
    ...     method='haversine', output_unit='km'
    ... )
    >>> gdt.fit(X)
    >>> X = gdt.transform(X)
    >>> X
       origin_lat  origin_lon  dest_lat   dest_lon  geo_distance
    0     40.7128    -74.0060   34.0522  -118.2437   3935.746254
    1     34.0522   -118.2437   41.8781   -87.6298   2808.517344
    2     41.8781    -87.6298   40.7128   -74.0060   1144.286561
    """

    def __init__(
        self,
        lat1: str,
        lon1: str,
        lat2: str,
        lon2: str,
        method: Literal["haversine", "euclidean", "manhattan"] = "haversine",
        output_unit: Literal["km", "miles", "meters", "feet"] = "km",
        output_col: str = "geo_distance",
        drop_original: bool = False,
    ) -> None:

        # Validate coordinate column names
        for param_name, param_value in [
            ("lat1", lat1),
            ("lon1", lon1),
            ("lat2", lat2),
            ("lon2", lon2),
        ]:
            if not isinstance(param_value, str):
                raise ValueError(
                    f"{param_name} must be a string. Got {type(param_value).__name__}."
                )

        # Validate method
        valid_methods = ["haversine", "euclidean", "manhattan"]
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}. Got '{method}' instead."
            )

        # Validate output_unit
        valid_units = ["km", "miles", "meters", "feet"]
        if output_unit not in valid_units:
            raise ValueError(
                f"output_unit must be one of {valid_units}. "
                f"Got '{output_unit}' instead."
            )

        # Validate output_col
        if not isinstance(output_col, str):
            raise ValueError(
                f"output_col must be a string. Got {type(output_col).__name__}."
            )

        _check_param_drop_original(drop_original)

        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2
        self.method = method
        self.output_unit = output_unit
        self.output_col = output_col
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Validates that the coordinate columns exist and are numerical.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.

        Returns
        -------
        self: GeoDistanceTransformer
            The fitted transformer.
        """

        # check input dataframe
        X = check_X(X)

        # Store coordinate variables
        self.variables_ = [self.lat1, self.lon1, self.lat2, self.lon2]

        # Check all coordinate columns exist
        missing = set(self.variables_) - set(X.columns)
        if missing:
            raise ValueError(
                f"Coordinate columns {missing} are not present in the dataframe."
            )

        # Check coordinate columns are numerical
        check_numerical_variables(X, self.variables_)

        # Check for missing values
        _check_contains_na(X, self.variables_)

        # Validate coordinate ranges (optional sanity check)
        for lat_col in [self.lat1, self.lat2]:
            if (X[lat_col].abs() > 90).any():
                raise ValueError(
                    f"Latitude values in '{lat_col}' must be between -90 and 90."
                )

        for lon_col in [self.lon1, self.lon2]:
            if (X[lon_col].abs() > 180).any():
                raise ValueError(
                    f"Longitude values in '{lon_col}' must be between -180 and 180."
                )

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate distances and add them as a new column.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe
            The dataframe with the new distance column added.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # Check for missing values
        _check_contains_na(X, self.variables_)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        # Calculate distance based on method
        if self.method == "haversine":
            distances = self._haversine_distance(
                X[self.lat1].values,
                X[self.lon1].values,
                X[self.lat2].values,
                X[self.lon2].values,
            )
        elif self.method == "euclidean":
            distances = self._euclidean_distance(
                X[self.lat1].values,
                X[self.lon1].values,
                X[self.lat2].values,
                X[self.lon2].values,
            )
        else:  # manhattan
            distances = self._manhattan_distance(
                X[self.lat1].values,
                X[self.lon1].values,
                X[self.lat2].values,
                X[self.lon2].values,
            )

        X[self.output_col] = distances

        if self.drop_original:
            X = X.drop(columns=self.variables_)

        return X

    def _haversine_distance(
        self,
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> np.ndarray:
        """Calculate the great-circle distance using the Haversine formula."""

        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        # Distance in the requested unit
        distance = EARTH_RADIUS[self.output_unit] * c

        return distance

    def _euclidean_distance(
        self,
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> np.ndarray:
        """Calculate Euclidean distance in coordinate space."""

        # Simple Euclidean distance (approximate, best for short distances)
        # Convert to approximate km then to requested unit
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Approximate degrees to km (at equator)
        km_per_degree = 111.0
        distance_km = np.sqrt((dlat * km_per_degree) ** 2 + (dlon * km_per_degree) ** 2)

        # Convert to requested unit
        conversion = EARTH_RADIUS[self.output_unit] / EARTH_RADIUS["km"]
        return distance_km * conversion

    def _manhattan_distance(
        self,
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> np.ndarray:
        """Calculate Manhattan (taxicab) distance in coordinate space."""

        dlat = np.abs(lat2 - lat1)
        dlon = np.abs(lon2 - lon1)

        # Approximate degrees to km (at equator)
        km_per_degree = 111.0
        distance_km = (dlat + dlon) * km_per_degree

        # Convert to requested unit
        conversion = EARTH_RADIUS[self.output_unit] / EARTH_RADIUS["km"]
        return distance_km * conversion

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. If None, uses ``feature_names_in_``.

        Returns
        -------
        feature_names_out : list of str
            Output feature names.
        """
        check_is_fitted(self)

        if self.drop_original:
            feature_names = [
                f for f in self.feature_names_in_ if f not in self.variables_
            ]
        else:
            feature_names = list(self.feature_names_in_)

        feature_names.append(self.output_col)

        return feature_names

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # This transformer has mandatory parameters
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has mandatory parameters"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
