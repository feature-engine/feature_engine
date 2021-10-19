CyclicalTransformer
===================

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.creation import CyclicalTransformer

    df = pd.DataFrame({
        'day': [6, 7, 5, 3, 1, 2, 4],
        'months': [3, 7, 9, 12, 4, 6, 12],
        })

    cyclical = CyclicalTransformer(variables=None, drop_original=True)

    X = cyclical.fit_transform(df)

.. code:: python

    print(cyclical.max_values_)

.. code:: python

    {'day': 7, 'months': 12}

.. code:: python

    print(X.head())

.. code:: python

          day_sin     day_cos  months_sin  months_cos
    1    -0.78183	  0.62349	      1.0	      0.0
    2         0.0	      1.0	     -0.5	 -0.86603
    3    -0.97493	-0.222521	     -1.0	     -0.0
    4     0.43388	-0.900969	      0.0	      1.0
    5     0.78183	  0.62349	  0.86603	     -0.5
    6     0.97493	-0.222521	      0.0	     -1.0
    7    -0.43388	-0.900969	      0.0	      1.0





