.. _geo_distance_transformer:

.. currentmodule:: feature_engine.creation

GeoDistanceFeatures
===================

:class:`GeoDistanceFeatures()` calculates the distance between two geographical
coordinate pairs (latitude/longitude) and adds the result as a new feature.

:class:`GeoDistanceFeatures()` is useful for location-based machine learning problems such as
real estate pricing, delivery route optimisation, ride-sharing applications,
and any domain where geographic proximity is relevant.

Distance methods
----------------

The transformer supports different distance calculation methods:

- **haversine**: Great-circle distance using the Haversine formula (default).
  Most accurate for typical distances on Earth's surface.
- **euclidean**: Simple Euclidean distance in the coordinate space.
  Fast but less accurate for long distances.
- **manhattan**: Manhattan (taxicab) distance in coordinate space.
  Useful as a rough approximation for grid-based city layouts.

Output units
------------

The distance can be returned in various units:

- **km**: Kilometres (default)
- **miles**: Miles
- **meters**: Metres
- **feet**: Feet

Python implementation
---------------------
Let's create a dataframe with origin and destination coordinates:

.. code:: python

    import pandas as pd
    from feature_engine.creation import GeoDistanceFeatures

    # Sample data: trips between US cities
    X = pd.DataFrame({
        'origin_lat': [40.7128, 34.0522, 41.8781, 29.7604],
        'origin_lon': [-74.0060, -118.2437, -87.6298, -95.3698],
        'dest_lat': [34.0522, 41.8781, 40.7128, 33.4484],
        'dest_lon': [-118.2437, -87.6298, -74.0060, -112.0740],
        'trip_id': [1, 2, 3, 4]
    })

Now let's calculate the distances using the haversine formula and returning the values in km:

.. code:: python

    # Set up the transformer
    gdt = GeoDistanceFeatures(
        lat1='origin_lat',
        lon1='origin_lon',
        lat2='dest_lat',
        lon2='dest_lon',
        method='haversine',
        output_unit='km',
        output_col='distance_km'
    )

    # Fit and transform
    gdt.fit(X)
    X_transformed = gdt.transform(X)

    print(X_transformed[['trip_id', 'distance_km']])

In the following output we see the trip ID followed by the distance travelled in each trip:

.. code:: python

       trip_id  distance_km
    0        1  3935.746255
    1        2  2803.971507
    2        3  1144.291274
    3        4  1632.166882

Using different distance methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use the Euclidean distance method, which provides a faster but less accurate
calculation (suitable for short distances):

.. code:: python

    gdt_euclidean = GeoDistanceFeatures(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        method='euclidean',
        output_col='distance_euclidean'
    )

    gdt_euclidean.fit(X)
    X_euclidean = gdt_euclidean.transform(X)
    print(X_euclidean[['trip_id', 'distance_euclidean']])

The Euclidean distances differ from the Haversine values because they don't account
for Earth's curvature:

.. code:: python

       trip_id  distance_euclidean
    0        1         4965.730734
    1        2         3507.416606
    2        3         1517.763567
    3        4         1898.819227

Alternatively, we can use the Manhattan distance, which is useful for grid-based city layouts:

.. code:: python

    gdt_manhattan = GeoDistanceFeatures(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        method='manhattan',
        output_col='distance_manhattan'
    )

    gdt_manhattan.fit(X)
    X_manhattan = gdt_manhattan.transform(X)
    print(X_manhattan[['trip_id', 'distance_manhattan']])

The Manhattan distance sums the absolute differences in latitude and longitude:

.. code:: python

       trip_id  distance_manhattan
    0        1           5649.7113
    1        2           4266.8178
    2        3           1641.5901
    3        4           2263.5342

Using different output units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transformer supports returning distances in km (default), miles, metres, or feet.
Here we calculate distances in miles:

.. code:: python

    gdt = GeoDistanceFeatures(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        output_unit='miles',
        output_col='distance_miles'
    )

    gdt.fit(X)
    X_transformed = gdt.transform(X)
    print(X_transformed[['trip_id', 'distance_miles']])

The distances are now expressed in miles instead of kilometres:

.. code:: python

       trip_id  distance_miles
    0        1     2445.586607
    1        2     1742.326542
    2        3      711.037560
    3        4     1014.192788

Dropping original coordinate columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reduce the dimensionality of the output dataset, we can remove the original
coordinate columns after calculating the distance:

.. code:: python

    gdt = GeoDistanceFeatures(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        drop_original=True
    )

    gdt.fit(X)
    X_transformed = gdt.transform(X)

    # Coordinate columns are removed
    print(X_transformed.columns.tolist())

After transformation, only the non-coordinate columns and the new distance column remain:

.. code:: python

    ['trip_id', 'geo_distance']

With polars
-----------

:class:`GeoDistanceFeatures()` works in the same way with a polars dataframe.
Let's create an equivalent toy dataset:

.. code:: python

    import polars as pl
    from feature_engine.creation import GeoDistanceFeatures

    X = pl.DataFrame({
        'origin_lat': [40.7128, 34.0522, 41.8781, 29.7604],
        'origin_lon': [-74.0060, -118.2437, -87.6298, -95.3698],
        'dest_lat': [34.0522, 41.8781, 40.7128, 33.4484],
        'dest_lon': [-118.2437, -87.6298, -74.0060, -112.0740],
        'trip_id': [1, 2, 3, 4]
    })

    gdt = GeoDistanceFeatures(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        method='haversine', output_unit='km', output_col='distance_km'
    )

    gdt.fit(X)
    X_transformed = gdt.transform(X)

    print(X_transformed.select(['trip_id', 'distance_km']))

We see the resulting distances:

.. code:: text

    shape: (4, 2)
    ┌─────────┬─────────────┐
    │ trip_id ┆ distance_km │
    │ ---     ┆ ---         │
    │ i64     ┆ f64         │
    ╞═════════╪═════════════╡
    │ 1       ┆ 3935.746255 │
    │ 2       ┆ 2803.971507 │
    │ 3       ┆ 1144.291274 │
    │ 4       ┆ 1632.166882 │
    └─────────┴─────────────┘

`drop_original=True` and the different distance methods and output units
work identically to the pandas examples above.

Calculating distance within a Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`GeoDistanceFeatures()` works seamlessly with scikit-learn pipelines. In the
following example, we create a pipeline that first calculates the geographic distance,
then scales the features, and finally trains a regression model:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    # Create sample target variable
    y = pd.Series([100, 150, 80, 200])

    # Create a pipeline for price prediction
    pipe = Pipeline([
        ('geo_distance', GeoDistanceFeatures(
            lat1='origin_lat', lon1='origin_lon',
            lat2='dest_lat', lon2='dest_lon',
            output_unit='km',
            drop_original=True
        )),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Fit the pipeline
    pipe.fit(X, y)

    # Make predictions
    predictions = pipe.predict(X)
    print(f"Predictions: {predictions}")

The pipeline successfully trains and returns predictions:

.. code:: python

    Predictions: [116.67298659 120.75252844  88.47598336 204.09850161]

Additional resources
--------------------

For tutorials about this and other feature engineering methods check out these resources:

- `Feature Engineering for Machine Learning <https://www.trainindata.com/p/feature-engineering-for-machine-learning>`_, online course.
- `Feature Engineering for Time Series Forecasting <https://www.trainindata.com/p/feature-engineering-for-forecasting>`_, online course.
- `Python Feature Engineering Cookbook <https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587>`_, book.

Both our book and courses are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting `Sole <https://linkedin.com/in/soledad-galli>`_,
the main developer of feature-engine.