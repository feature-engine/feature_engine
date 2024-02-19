from typing import Optional, Union

import numpy as np
import pandas as pd

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._base_transformers.mixins import (
    FitFromDictMixin,
    GetFeatureNamesOutMixin,
)
from feature_engine._check_input_parameters.check_init_input_params import (
    _check_param_drop_original,
)


class DistanceFeatures(
    BaseNumericalTransformer, FitFromDictMixin, GetFeatureNamesOutMixin
):
    EARTH_RADIUS: float = 6371.  # radius of Earth in kms

    def __init__(
            self,
            a_latitude: str,
            a_longitude: str,
            b_latitude: str,
            b_longitude: str,
            output_column_name: Union[str, None] = None,
            drop_original: bool = False,
    ) -> None:

        self.a_latitude = self._check_column_name(a_latitude)
        self.a_longitude = self._check_column_name(a_longitude)
        self.b_latitude = self._check_column_name(b_latitude)
        self.b_longitude = self._check_column_name(b_longitude)

        self.output_column_name = self._check_column_name(column_name=output_column_name)

        _check_param_drop_original(drop_original=drop_original)
        self.drop_original = drop_original

        self.variables = None

    @staticmethod
    def _check_column_name(column_name: str) -> str:
        if not isinstance(column_name, str):
            raise ValueError(
                "column_name takes only string as value. "
                f"Got {column_name} instead."
            )

        return column_name

    def transform(self, X: pd.DataFrame):
        """
        Compute the distance between the two coordinates given using the Haversine formula

        Parameters
        ----------
        X: Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: Pandas dataframe.
            The original dataframe plus the distance between the given coordinates.
        """
        X = self._check_transform_input_and_state(X)
        X = self._check_lat_lon_columns_are_in_df(X)
        X = self._check_correctness_of_coordinates(X)

        self.compute_distance(X)

        if self.drop_original:
            X.drop(
                columns=[
                    self.a_latitude,
                    self.a_longitude,
                    self.b_latitude,
                    self.b_longitude,
                ],
                inplace=True)

        return X

    def compute_distance(self, X: pd.DataFrame):
        # convert latitude and longitude in radians
        phi_1 = np.radians(X[self.a_latitude])
        phi_2 = np.radians(X[self.b_latitude])
        lambda_1 = np.radians(X[self.a_longitude])
        lambda_2 = np.radians(X[self.b_longitude])

        # compute delta, i.e., difference, between radians
        delta_phi = phi_2 - phi_1
        delta_lambda = lambda_2 - lambda_1

        # compute distance using Haversine formula
        inner_part = np.sin(delta_phi / 2) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
        X[self.output_column_name] = self.EARTH_RADIUS * 2 * np.arcsin(np.sqrt(inner_part))

    def _check_lat_lon_columns_are_in_df(self, X) -> pd.DataFrame:
        df_columns = set(X.columns)
        input_columns = {self.a_latitude, self.a_longitude, self.b_latitude, self.b_latitude}

        if input_columns.issubset(df_columns) is False:
            raise ValueError(f'The columns {input_columns.difference(df_columns)} were not found in the dataframe.')

        return X

    def _check_correctness_of_coordinates(self, X: pd.DataFrame) -> pd.DataFrame:
        irregular_latitudes = X[(X[self.a_latitude].abs() > 90) | (X[self.b_latitude].abs() > 90)]
        irregular_longitudes = X[(X[self.a_longitude].abs() > 180) | (X[self.b_longitude].abs() > 180)]

        if irregular_latitudes.empty is False:
            raise ValueError(f'The dataframe contains irregular latitudes: {irregular_latitudes}')
        if irregular_longitudes.empty is False:
            raise ValueError(f'The dataframe contains irregular longitudes: {irregular_longitudes}')

        return X
