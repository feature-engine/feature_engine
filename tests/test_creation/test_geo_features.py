import numpy as np
import pandas as pd
import pytest

from feature_engine.creation import GeoDistanceTransformer


@pytest.fixture
def df_coords():
    """Fixture providing sample coordinate data."""
    return pd.DataFrame(
        {
            "lat1": [40.7128],
            "lon1": [-74.0060],
            "lat2": [34.0522],
            "lon2": [-118.2437],
        }
    )


def test_haversine_distance_default(df_coords):
    """Test Haversine distance calculation with default parameters."""
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    X_tr = transformer.fit_transform(df_coords)

    # Distance should be approximately 3935-3944 km
    assert "geo_distance" in X_tr.columns
    assert 3900 < X_tr["geo_distance"].iloc[0] < 4000


def test_haversine_distance_miles():
    """Test Haversine distance in miles."""
    X = pd.DataFrame(
        {
            "lat1": [40.7128],
            "lon1": [-74.0060],
            "lat2": [34.0522],
            "lon2": [-118.2437],
        }
    )
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="miles"
    )
    X_tr = transformer.fit_transform(X)

    # Distance should be approximately 2445 miles
    assert 2400 < X_tr["geo_distance"].iloc[0] < 2500


def test_same_location_zero_distance():
    """Test that same location returns zero distance."""
    X = pd.DataFrame(
        {
            "lat1": [40.7128, 34.0522],
            "lon1": [-74.0060, -118.2437],
            "lat2": [40.7128, 34.0522],
            "lon2": [-74.0060, -118.2437],
        }
    )
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    X_tr = transformer.fit_transform(X)

    np.testing.assert_array_almost_equal(
        X_tr["geo_distance"].values, [0.0, 0.0], decimal=10
    )


def test_euclidean_method():
    """Test Euclidean distance method."""
    X = pd.DataFrame({"lat1": [0.0], "lon1": [0.0], "lat2": [1.0], "lon2": [1.0]})
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", method="euclidean"
    )
    X_tr = transformer.fit_transform(X)

    assert X_tr["geo_distance"].iloc[0] > 0


def test_manhattan_method():
    """Test Manhattan distance method."""
    X = pd.DataFrame({"lat1": [0.0], "lon1": [0.0], "lat2": [1.0], "lon2": [1.0]})
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", method="manhattan"
    )
    X_tr = transformer.fit_transform(X)

    assert X_tr["geo_distance"].iloc[0] > 0


def test_custom_output_column_name(df_coords):
    """Test custom output column name."""
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_col="distance_km"
    )
    X_tr = transformer.fit_transform(df_coords)

    assert "distance_km" in X_tr.columns
    assert "geo_distance" not in X_tr.columns


def test_drop_original_columns():
    """Test drop_original parameter."""
    X = pd.DataFrame(
        {
            "lat1": [40.7128],
            "lon1": [-74.0060],
            "lat2": [34.0522],
            "lon2": [-118.2437],
            "other": [1],
        }
    )
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", drop_original=True
    )
    X_tr = transformer.fit_transform(X)

    assert "lat1" not in X_tr.columns
    assert "lon1" not in X_tr.columns
    assert "lat2" not in X_tr.columns
    assert "lon2" not in X_tr.columns
    assert "geo_distance" in X_tr.columns
    assert "other" in X_tr.columns


def test_multiple_rows():
    """Test with multiple rows."""
    X = pd.DataFrame(
        {
            "origin_lat": [40.7128, 34.0522, 41.8781],
            "origin_lon": [-74.0060, -118.2437, -87.6298],
            "dest_lat": [34.0522, 41.8781, 40.7128],
            "dest_lon": [-118.2437, -87.6298, -74.0060],
        }
    )
    transformer = GeoDistanceTransformer(
        lat1="origin_lat", lon1="origin_lon", lat2="dest_lat", lon2="dest_lon"
    )
    X_tr = transformer.fit_transform(X)

    assert len(X_tr["geo_distance"]) == 3
    # All distances should be positive
    assert all(X_tr["geo_distance"] > 0)


def test_invalid_method_raises_error():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="method must be one of"):
        GeoDistanceTransformer(
            lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", method="invalid"
        )


def test_invalid_output_unit_raises_error():
    """Test that invalid output_unit raises ValueError."""
    with pytest.raises(ValueError, match="output_unit must be one of"):
        GeoDistanceTransformer(
            lat1="lat1",
            lon1="lon1",
            lat2="lat2",
            lon2="lon2",
            output_unit="invalid",
        )


def test_missing_columns_raises_error():
    """Test that missing columns raise ValueError on fit."""
    X = pd.DataFrame({"lat1": [1], "lon1": [1]})
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    with pytest.raises(ValueError, match="not present in the dataframe"):
        transformer.fit(X)


def test_invalid_latitude_range_raises_error():
    """Test that latitude out of range raises ValueError when validate_ranges=True."""
    X = pd.DataFrame(
        {
            "lat1": [100],  # Invalid: outside -90 to 90
            "lon1": [0],
            "lat2": [0],
            "lon2": [0],
        }
    )
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    with pytest.raises(ValueError, match="Latitude values.*must be between"):
        transformer.fit(X)


def test_invalid_longitude_range_raises_error():
    """Test that longitude out of range raises ValueError when validate_ranges=True."""
    X = pd.DataFrame(
        {
            "lat1": [0],
            "lon1": [200],  # Invalid: outside -180 to 180
            "lat2": [0],
            "lon2": [0],
        }
    )
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    with pytest.raises(ValueError, match="Longitude values.*must be between"):
        transformer.fit(X)


def test_validate_ranges_disabled():
    """Test that invalid coordinates don't raise error when validate_ranges=False."""
    X = pd.DataFrame(
        {
            "lat1": [100],  # Invalid latitude
            "lon1": [200],  # Invalid longitude
            "lat2": [0],
            "lon2": [0],
        }
    )
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", validate_ranges=False
    )
    # Should not raise an error
    transformer.fit(X)
    X_tr = transformer.transform(X)
    # Distance may be incorrect but should complete
    assert "geo_distance" in X_tr.columns


def test_validate_ranges_parameter_validation():
    """Test that validate_ranges must be boolean."""
    with pytest.raises(ValueError, match="validate_ranges must be a boolean"):
        GeoDistanceTransformer(
            lat1="lat1",
            lon1="lon1",
            lat2="lat2",
            lon2="lon2",
            validate_ranges="True",
        )


def test_fit_stores_attributes():
    """Test that fit stores expected attributes."""
    X = pd.DataFrame(
        {"lat1": [40.0], "lon1": [-74.0], "lat2": [34.0], "lon2": [-118.0]}
    )
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    transformer.fit(X)

    assert hasattr(transformer, "variables_")
    assert hasattr(transformer, "feature_names_in_")
    assert hasattr(transformer, "n_features_in_")
    assert set(transformer.variables_) == {"lat1", "lon1", "lat2", "lon2"}


def test_get_feature_names_out():
    """Test get_feature_names_out returns correct names."""
    X = pd.DataFrame(
        {
            "lat1": [40.0],
            "lon1": [-74.0],
            "lat2": [34.0],
            "lon2": [-118.0],
            "other": [1],
        }
    )
    transformer = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    transformer.fit(X)

    feature_names = transformer.get_feature_names_out()
    assert "geo_distance" in feature_names
    assert len(feature_names) == 6  # 5 original + 1 new


def test_output_units_conversion():
    """Test different output units give consistent results."""
    X = pd.DataFrame(
        {
            "lat1": [40.7128],
            "lon1": [-74.0060],
            "lat2": [34.0522],
            "lon2": [-118.2437],
        }
    )

    # Get distance in km and miles
    transformer_km = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="km"
    )
    transformer_miles = GeoDistanceTransformer(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="miles"
    )

    dist_km = transformer_km.fit_transform(X.copy())["geo_distance"].iloc[0]
    dist_miles = transformer_miles.fit_transform(X.copy())["geo_distance"].iloc[0]

    # 1 km ≈ 0.621371 miles
    expected_miles = dist_km * 0.621371
    np.testing.assert_almost_equal(dist_miles, expected_miles, decimal=0)
