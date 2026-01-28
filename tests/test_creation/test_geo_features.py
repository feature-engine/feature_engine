import numpy as np
import pandas as pd
import pytest

from feature_engine.creation import GeoDistanceFeatures


@pytest.fixture
def df_coords():
    """Fixture providing sample coordinate data for a single route."""
    return pd.DataFrame(
        {
            "lat1": [40.7128],
            "lon1": [-74.0060],
            "lat2": [34.0522],
            "lon2": [-118.2437],
        }
    )


@pytest.fixture
def df_multi_coords():
    """Fixture providing sample coordinate data with multiple rows."""
    return pd.DataFrame(
        {
            "origin_lat": [40.7128, 34.0522, 41.8781],
            "origin_lon": [-74.0060, -118.2437, -87.6298],
            "dest_lat": [34.0522, 41.8781, 40.7128],
            "dest_lon": [-118.2437, -87.6298, -74.0060],
        }
    )


@pytest.fixture
def df_with_extra():
    """Fixture for DataFrame with coordinates and extra columns."""
    return pd.DataFrame(
        {
            "lat1": [40.0],
            "lon1": [-74.0],
            "lat2": [34.0],
            "lon2": [-118.0],
            "other": [1],
        }
    )


def test_haversine_distance_default(df_coords):
    """Test Haversine distance calculation with default parameters."""
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    X_tr = transformer.fit_transform(df_coords)

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
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="miles"
    )
    X_tr = transformer.fit_transform(X)

    assert 2400 < X_tr["geo_distance"].iloc[0] < 2500


@pytest.mark.parametrize("method", ["haversine", "euclidean", "manhattan"])
@pytest.mark.parametrize("output_unit", ["km", "miles", "meters", "feet"])
def test_same_location_zero_distance(method, output_unit):
    """Test that same location returns zero distance for all methods and units."""
    X = pd.DataFrame(
        {
            "lat1": [40.7128, 34.0522],
            "lon1": [-74.0060, -118.2437],
            "lat2": [40.7128, 34.0522],
            "lon2": [-74.0060, -118.2437],
        }
    )
    transformer = GeoDistanceFeatures(
        lat1="lat1",
        lon1="lon1",
        lat2="lat2",
        lon2="lon2",
        method=method,
        output_unit=output_unit,
    )
    X_tr = transformer.fit_transform(X)

    np.testing.assert_array_almost_equal(
        X_tr["geo_distance"].values, [0.0, 0.0], decimal=10
    )


def test_euclidean_method():
    """Test Euclidean distance method returns expected values."""
    X = pd.DataFrame({"lat1": [0.0], "lon1": [0.0], "lat2": [1.0], "lon2": [1.0]})
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", method="euclidean"
    )
    X_tr = transformer.fit_transform(X)

    expected_distance = np.sqrt(2) * 111.0
    np.testing.assert_almost_equal(
        X_tr["geo_distance"].iloc[0], expected_distance, decimal=1
    )


def test_manhattan_method():
    """Test Manhattan distance method returns expected values."""
    X = pd.DataFrame({"lat1": [0.0], "lon1": [0.0], "lat2": [1.0], "lon2": [1.0]})
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", method="manhattan"
    )
    X_tr = transformer.fit_transform(X)

    expected_distance = 2 * 111.0
    np.testing.assert_almost_equal(
        X_tr["geo_distance"].iloc[0], expected_distance, decimal=1
    )


def test_custom_output_column_name(df_coords):
    """Test custom output column name."""
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_col="distance_km"
    )
    X_tr = transformer.fit_transform(df_coords)

    assert "distance_km" in X_tr.columns
    assert "geo_distance" not in X_tr.columns


def test_drop_original_columns():
    """Test drop_original parameter removes coordinate columns."""
    X = pd.DataFrame(
        {
            "lat1": [40.7128],
            "lon1": [-74.0060],
            "lat2": [34.0522],
            "lon2": [-118.2437],
            "other": [1],
        }
    )
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", drop_original=True
    )
    X_tr = transformer.fit_transform(X)

    assert "lat1" not in X_tr.columns
    assert "lon1" not in X_tr.columns
    assert "lat2" not in X_tr.columns
    assert "lon2" not in X_tr.columns
    assert "geo_distance" in X_tr.columns
    assert "other" in X_tr.columns
    assert list(X_tr.columns) == ["other", "geo_distance"]


def test_multiple_rows(df_multi_coords):
    """Test transformation with multiple rows returns expected distances."""
    transformer = GeoDistanceFeatures(
        lat1="origin_lat", lon1="origin_lon", lat2="dest_lat", lon2="dest_lon"
    )
    X_tr = transformer.fit_transform(df_multi_coords)

    expected = df_multi_coords.copy()
    expected["geo_distance"] = [
        3935.746254609723,
        2803.971506975193,
        1144.2912739463475,
    ]

    pd.testing.assert_frame_equal(
        X_tr,
        expected,
        check_exact=False,
        atol=0.001,
    )


@pytest.mark.parametrize("invalid_method", ["invalid", True, 123])
def test_invalid_method_raises_error(invalid_method):
    """Test that invalid method values raise ValueError."""
    with pytest.raises(ValueError, match="method must be one of"):
        GeoDistanceFeatures(
            lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", method=invalid_method
        )


@pytest.mark.parametrize("invalid_unit", ["invalid", True, 123])
def test_invalid_output_unit_raises_error(invalid_unit):
    """Test that invalid output_unit values raise ValueError."""
    with pytest.raises(ValueError, match="output_unit must be one of"):
        GeoDistanceFeatures(
            lat1="lat1",
            lon1="lon1",
            lat2="lat2",
            lon2="lon2",
            output_unit=invalid_unit,
        )


def test_missing_columns_raises_error():
    """Test that missing columns raise ValueError on fit."""
    X = pd.DataFrame({"lat1": [1], "lon1": [1]})
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    with pytest.raises(ValueError, match="not present in the dataframe"):
        transformer.fit(X)


@pytest.mark.parametrize("invalid_lat", [100, -100])
def test_invalid_latitude_range_raises_error(invalid_lat):
    """Test that latitude outside [-90, 90] raises ValueError."""
    X = pd.DataFrame(
        {
            "lat1": [invalid_lat],
            "lon1": [0],
            "lat2": [0],
            "lon2": [0],
        }
    )
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    with pytest.raises(ValueError, match="Latitude values.*must be between"):
        transformer.fit(X)


@pytest.mark.parametrize("invalid_lon", [200, -200])
def test_invalid_longitude_range_raises_error(invalid_lon):
    """Test that longitude outside [-180, 180] raises ValueError."""
    X = pd.DataFrame(
        {
            "lat1": [0],
            "lon1": [invalid_lon],
            "lat2": [0],
            "lon2": [0],
        }
    )
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    with pytest.raises(ValueError, match="Longitude values.*must be between"):
        transformer.fit(X)


def test_validate_ranges_disabled():
    """Test that invalid coordinates don't raise error when validate_ranges=False."""
    X = pd.DataFrame(
        {
            "lat1": [100],
            "lon1": [200],
            "lat2": [0],
            "lon2": [0],
        }
    )
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", validate_ranges=False
    )
    transformer.fit(X)
    X_tr = transformer.transform(X)
    assert "geo_distance" in X_tr.columns


@pytest.mark.parametrize("invalid_value", ["True", 123, 0.5])
def test_validate_ranges_parameter_validation(invalid_value):
    """Test that validate_ranges must be a boolean."""
    with pytest.raises(ValueError, match="validate_ranges must be a boolean"):
        GeoDistanceFeatures(
            lat1="lat1",
            lon1="lon1",
            lat2="lat2",
            lon2="lon2",
            validate_ranges=invalid_value,
        )


def test_fit_stores_attributes():
    """Test that fit stores expected attributes with correct values."""
    X = pd.DataFrame(
        {"lat1": [40.0], "lon1": [-74.0], "lat2": [34.0], "lon2": [-118.0]}
    )
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    transformer.fit(X)

    assert hasattr(transformer, "variables_")
    assert hasattr(transformer, "feature_names_in_")
    assert hasattr(transformer, "n_features_in_")
    assert set(transformer.variables_) == {"lat1", "lon1", "lat2", "lon2"}
    assert transformer.feature_names_in_ == ["lat1", "lon1", "lat2", "lon2"]
    assert transformer.n_features_in_ == 4


def test_get_feature_names_out(df_with_extra):
    """Test get_feature_names_out returns correct feature names."""
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    transformer.fit(df_with_extra)

    feature_names = transformer.get_feature_names_out()
    expected_names = ["lat1", "lon1", "lat2", "lon2", "other", "geo_distance"]
    assert feature_names == expected_names


def test_get_feature_names_out_with_drop_original(df_with_extra):
    """Test get_feature_names_out when drop_original=True."""
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", drop_original=True
    )
    transformer.fit(df_with_extra)

    feature_names = transformer.get_feature_names_out()
    expected_names = ["other", "geo_distance"]
    assert feature_names == expected_names


def test_output_units_conversion():
    """Test different output units give consistent results with correct conversion."""
    X = pd.DataFrame(
        {
            "lat1": [40.7128],
            "lon1": [-74.0060],
            "lat2": [34.0522],
            "lon2": [-118.2437],
        }
    )

    transformer_km = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="km"
    )
    transformer_miles = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="miles"
    )

    dist_km = transformer_km.fit_transform(X.copy())["geo_distance"].iloc[0]
    dist_miles = transformer_miles.fit_transform(X.copy())["geo_distance"].iloc[0]

    expected_miles = dist_km * 0.621371
    np.testing.assert_almost_equal(dist_miles, expected_miles, decimal=0)


def test_invalid_param_types_raises_error():
    """Test that invalid parameter types raise ValueError."""
    # Test lat1 not string
    with pytest.raises(ValueError, match="lat1 must be a string"):
        GeoDistanceFeatures(lat1=123, lon1="lon1", lat2="lat2", lon2="lon2")

    # Test output_col not string
    with pytest.raises(ValueError, match="output_col must be a string"):
        GeoDistanceFeatures(
            lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_col=123
        )


def test_more_tags_and_sklearn_tags():
    """Test that _more_tags returns expected dictionary."""
    transformer = GeoDistanceFeatures(lat1="l1", lon1="lg1", lat2="l2", lon2="lg2")
    tags = transformer._more_tags()
    assert tags["variables"] == "numerical"
    assert (
        tags["_xfail_checks"]["check_parameters_default_constructible"]
        == "transformer has mandatory parameters"
    )

    # basic check for sklearn tags if available (new sklearn versions)
    if hasattr(transformer, "__sklearn_tags__"):
        tags = transformer.__sklearn_tags__()
        assert tags is not None
