SklearnTransformerWrapper
=========================

API Reference
-------------

.. autoclass:: feature_engine.wrappers.SklearnTransformerWrapper
    :members:

Example
-------

Implements Scikit-learn transformers like the SimpleImputer, the OrdinalEncoder or most
scalers only to the selected subset of features.

In the next code snippet we show how to wrap the SimpleImputer from Scikit-learn to
impute only the selected variables.

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from feature_engine.wrappers import SklearnTransformerWrapper
	
    # Load dataset
    data = pd.read_csv('houseprice.csv')
    
    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1),
    	data['SalePrice'], test_size=0.3, random_state=0)
    	
    # set up the wrapper with the SimpleImputer
    imputer = SklearnTransformerWrapper(transformer = SimpleImputer(strategy='mean'),
                                        variables = ['LotFrontage', 'MasVnrArea'])
    
    # fit the wrapper + SimpleImputer                              
    imputer.fit(X_train)
	
    # transform the data
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)


In the next snippet of code we show how to wrap the StandardScaler from Scikit-learn
to standardize only the selected variables.

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from feature_engine.wrappers import SklearnTransformerWrapper

    # Load dataset
    data = pd.read_csv('houseprice.csv')

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1),
    	data['SalePrice'], test_size=0.3, random_state=0)

    # set up the wrapper with the StandardScaler
    scaler = SklearnTransformerWrapper(transformer = StandardScaler(),
                                        variables = ['LotFrontage', 'MasVnrArea'])

    # fit the wrapper + StandardScaler
    scaler.fit(X_train)

    # transform the data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


In the next snippet of code we show how to wrap the SelectKBest from Scikit-learn
to select only a subset of the variables.

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import f_regression, SelectKBest
    from feature_engine.wrappers import SklearnTransformerWrapper

    # Load dataset
    data = pd.read_csv('houseprice.csv')

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1),
    	data['SalePrice'], test_size=0.3, random_state=0)

    cols = [var for var in X_train.columns if X_train[var].dtypes !='O']

    # let's apply the standard scaler on the above variables

    selector = SklearnTransformerWrapper(
        transformer = SelectKBest(f_regression, k=5),
        variables = cols)

    selector.fit(X_train.fillna(0), y_train)

    # transform the data
    X_train_t = selector.transform(X_train.fillna(0))
    X_test_t = selector.transform(X_test.fillna(0))
