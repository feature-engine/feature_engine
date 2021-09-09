MatchColumnsToTrainSet
======================

API Reference
-------------

.. autoclass:: feature_engine.preprocessing.MatchColumnsToTrainSet
    :members:


Example
-------

The MatchColumnsToTrainSet() ensure that columns in test dataset are similar 
to train dataset. 
If needed it drops columns that are in test but not in the train dataset
and add columns that are missing 
in test dataset (but are in train).

.. code:: python

    import pandas as pd

    from feature_engine.preprocessing import MatchColumnsToTrainSet


    # Load dataset
    def load_titanic():
            data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
            data = data.replace('?', np.nan)
            data['cabin'] = data['cabin'].astype(str).str[0]
            data['pclass'] = data['pclass'].astype('O')
            data['embarked'].fillna('C', inplace=True)
            return data

    # load data as pandas dataframe
    data = load_titanic()

    # Split test and train
    train = data.iloc[0:1000, :]
    test = data.iloc[1000:, :]

    columns_matcher = MatchColumnsToTrainSet(missing_values="ignore")

    # We drop some columns to test the transformer
    test_with_missing_columns = test.drop(["sex", "ticket"], axis=1)

    test_with_missing_columns.head()

.. code:: python

    pclass	survived	fare	cabin	embarked
    1000	3	1	7.7500	n	Q
    1001	3	1	23.2500	n	Q
    1002	3	1	23.2500	n	Q
    1003	3	1	23.2500	n	Q
    1004	3	1	7.7875	n	Q

.. code:: python

    columns_matcher.fit(train)

    res = columns_matcher.transform(test_with_missing_columns)

    res.head()

.. code:: python

    pclass	survived	sex	ticket	fare	cabin	embarked
    1000	3	1	NaN	NaN	7.7500	n	Q
    1001	3	1	NaN	NaN	23.2500	n	Q
    1002	3	1	NaN	NaN	23.2500	n	Q
    1003	3	1	NaN	NaN	23.2500	n	Q
    1004	3	1	NaN	NaN	7.7875	n	Q


Missing columns were created with missing values.
