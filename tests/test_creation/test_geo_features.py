import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.creation import GeoDistanceFeatures

COORDS_DATA = {
    "lat1": [40.7128],
    "lon1": [-74.0060],
    "lat2": [34.0522],
    "lon2": [-118.2437],
}

MULTI_COORDS_DATA = {
    "origin_lat": [40.7128, 34.0522, 41.8781],
    "origin_lon": [-74.0060, -118.2437, -87.6298],
    "dest_lat": [34.0522, 41.8781, 40.7128],
    "dest_lon": [-118.2437, -87.6298, -74.0060],
}

COORDS_WITH_EXTRA_DATA = {
    "lat1": [40.0],
    "lon1": [-74.0],
    "lat2": [34.0],
    "lon2": [-118.0],
    "other": [1],
}


def get_value(X, col: str, idx: int = 0):
    """Extract a single scalar from a pandas or polars dataframe column."""
    return nw.from_native(X, eager_only=True).get_column(col).to_list()[idx]


def assert_df_equal(X, expected: dict, abs_tol: float = 1e-5) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert result[col] == pytest.approx(values, abs=abs_tol)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_haversine_distance_default(make_df):
    """Test Haversine distance calculation with default parameters."""
    df = make_df(COORDS_DATA)
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    X_tr = transformer.fit_transform(df)

    assert "geo_distance" in X_tr.columns
    assert 3900 < get_value(X_tr, "geo_distance") < 4000


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_haversine_distance_miles(make_df):
    """Test Haversine distance in miles."""
    X = make_df(COORDS_DATA)
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="miles"
    )
    X_tr = transformer.fit_transform(X)

    assert 2400 < get_value(X_tr, "geo_distance") < 2500


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("method", ["haversine", "euclidean", "manhattan"])
@pytest.mark.parametrize("output_unit", ["km", "miles", "meters", "feet"])
def test_same_location_zero_distance(make_df, method, output_unit):
    """Test that same location returns zero distance for all methods and units."""
    X = make_df(
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

    values = nw.from_native(X_tr, eager_only=True).get_column("geo_distance")
    np.testing.assert_array_almost_equal(values.to_list(), [0.0, 0.0], decimal=10)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_euclidean_method(make_df):
    """Test Euclidean distance method returns expected values."""
    X = make_df({"lat1": [0.0], "lon1": [0.0], "lat2": [1.0], "lon2": [1.0]})
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", method="euclidean"
    )
    X_tr = transformer.fit_transform(X)

    expected_distance = np.sqrt(2) * 111.0
    np.testing.assert_almost_equal(
        get_value(X_tr, "geo_distance"), expected_distance, decimal=1
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_manhattan_method(make_df):
    """Test Manhattan distance method returns expected values."""
    X = make_df({"lat1": [0.0], "lon1": [0.0], "lat2": [1.0], "lon2": [1.0]})
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", method="manhattan"
    )
    X_tr = transformer.fit_transform(X)

    expected_distance = 2 * 111.0
    np.testing.assert_almost_equal(
        get_value(X_tr, "geo_distance"), expected_distance, decimal=1
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_custom_output_column_name(make_df):
    """Test custom output column name."""
    df = make_df(COORDS_DATA)
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_col="distance_km"
    )
    X_tr = transformer.fit_transform(df)

    assert "distance_km" in X_tr.columns
    assert "geo_distance" not in X_tr.columns


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_drop_original_columns(make_df):
    """Test drop_original parameter removes coordinate columns."""
    X = make_df(
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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_multiple_rows(make_df):
    """Test transformation with multiple rows returns expected distances."""
    df = make_df(MULTI_COORDS_DATA)
    transformer = GeoDistanceFeatures(
        lat1="origin_lat", lon1="origin_lon", lat2="dest_lat", lon2="dest_lon"
    )
    X_tr = transformer.fit_transform(df)

    expected = dict(MULTI_COORDS_DATA)
    expected["geo_distance"] = [
        3935.746254609723,
        2803.971506975193,
        1144.2912739463475,
    ]

    assert_df_equal(X_tr, expected, abs_tol=0.001)


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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_missing_columns_raises_error(make_df):
    """Test that missing columns raise ValueError on fit."""
    X = make_df({"lat1": [1], "lon1": [1]})
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    with pytest.raises(ValueError, match="not present in the dataframe"):
        transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("invalid_lat", [100, -100])
def test_invalid_latitude_range_raises_error(make_df, invalid_lat):
    """Test that latitude outside [-90, 90] raises ValueError."""
    X = make_df(
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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("invalid_lon", [200, -200])
def test_invalid_longitude_range_raises_error(make_df, invalid_lon):
    """Test that longitude outside [-180, 180] raises ValueError."""
    X = make_df(
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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_validate_ranges_disabled(make_df):
    """Test that invalid coordinates don't raise error when validate_ranges=False."""
    X = make_df(
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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_stores_attributes(make_df):
    """Test that fit stores expected attributes with correct values."""
    X = make_df({"lat1": [40.0], "lon1": [-74.0], "lat2": [34.0], "lon2": [-118.0]})
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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out(make_df):
    """Test get_feature_names_out returns correct feature names."""
    df = make_df(COORDS_WITH_EXTRA_DATA)
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2"
    )
    transformer.fit(df)

    feature_names = transformer.get_feature_names_out()
    expected_names = ["lat1", "lon1", "lat2", "lon2", "other", "geo_distance"]
    assert feature_names == expected_names


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out_with_drop_original(make_df):
    """Test get_feature_names_out when drop_original=True."""
    df = make_df(COORDS_WITH_EXTRA_DATA)
    transformer = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", drop_original=True
    )
    transformer.fit(df)

    feature_names = transformer.get_feature_names_out()
    expected_names = ["other", "geo_distance"]
    assert feature_names == expected_names


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_output_units_conversion(make_df):
    """Test different output units give consistent results with correct conversion."""
    data = COORDS_DATA

    transformer_km = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="km"
    )
    transformer_miles = GeoDistanceFeatures(
        lat1="lat1", lon1="lon1", lat2="lat2", lon2="lon2", output_unit="miles"
    )

    dist_km = get_value(transformer_km.fit_transform(make_df(data)), "geo_distance")
    dist_miles = get_value(
        transformer_miles.fit_transform(make_df(data)), "geo_distance"
    )

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

    tags = transformer.__sklearn_tags__()
    assert tags is not None
