.. _sklearn_wrapper:

.. currentmodule:: feature_engine.wrappers

SklearnTransformerWrapper
=========================

The :class:`SklearnTransformerWrapper()` applies scikit-learn transformers to a selected
group of variables. It works with transformers like the SimpleImputer, OrdinalEncoder,
OneHotEncoder, KBinsDiscretizer, all scalers and also transformers for feature selection.
Other transformers have not been tested, but we think it should work with most of them.

The :class:`SklearnTransformerWrapper()` offers similar functionality to the
`ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_
class available in scikit-learn. They differ in the implementation to select the
variables and the output.

The :class:`SklearnTransformerWrapper()` returns a pandas dataframe with the variables
in the order of the original data. The
`ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_
returns a Numpy array, and the order of the variables may not coincide with that of the
original dataset.

.. note::

    **New in version 2.0:** When `variables` is `None`, :class:`SklearnTransformerWrapper()`
    used to raise an error if the dataframe contained no variables of the relevant
    type. You can now set the new parameter `return_empty` to `True` to make the
    transformer return an empty list of variables and skip the transformation
    instead, leaving the dataframe unchanged. This lets you reuse the same pipeline
    across different datasets or projects, some of which may not contain variables
    of the relevant type, without building a tailored pipeline for each one.
    `return_empty` will default to `True` from version 2.1 onwards.

In the next code snippet we show how to wrap the SimpleImputer from scikit-learn to
impute only the selected variables. We start with the imports:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml
    from sklearn.impute import SimpleImputer
    from feature_engine.wrappers import SklearnTransformerWrapper

Next, we load the house prices dataset and split it into a train set and a test set:

.. code:: python

    # Load dataset
    data = fetch_openml(
        name='house_prices',
        version=1,
        as_frame=True,
        parser='auto',
    ).frame

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Id', 'SalePrice'], axis=1),
        data['SalePrice'], test_size=0.3, random_state=0)

Now, we set up the wrapper with the SimpleImputer, and fit it to the train set:

.. code:: python

    # set up the wrapper with the SimpleImputer
    imputer = SklearnTransformerWrapper(transformer = SimpleImputer(strategy='mean'),
                                        variables = ['LotFrontage', 'MasVnrArea'])

    # fit the wrapper + SimpleImputer
    imputer.fit(X_train)

Finally, we impute the missing data in the train and test sets:

.. code:: python

    # transform the data
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)


In the next snippet of code we show how to wrap the StandardScaler from scikit-learn
to standardise only the selected variables. We start with the imports:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from feature_engine.wrappers import SklearnTransformerWrapper

Next, we load the house prices dataset and split it into a train set and a test set:

.. code:: python

    # Load dataset
    data = fetch_openml(
        name='house_prices',
        version=1,
        as_frame=True,
        parser='auto',
    ).frame

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Id', 'SalePrice'], axis=1),
        data['SalePrice'], test_size=0.3, random_state=0)

Now, we set up the wrapper with the StandardScaler, and fit it to the train set:

.. code:: python

    # set up the wrapper with the StandardScaler
    scaler = SklearnTransformerWrapper(transformer = StandardScaler(),
                                        variables = ['LotFrontage', 'MasVnrArea'])

    # fit the wrapper + StandardScaler
    scaler.fit(X_train)

Finally, we standardise the selected variables in the train and test sets:

.. code:: python

    # transform the data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


In the next snippet of code we show how to wrap the SelectKBest from scikit-learn
to select only a subset of the variables. We start with the imports:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml
    from sklearn.feature_selection import f_regression, SelectKBest
    from feature_engine.wrappers import SklearnTransformerWrapper

Next, we load the house prices dataset and split it into a train set and a test set:

.. code:: python

    # Load dataset
    data = fetch_openml(
        name='house_prices',
        version=1,
        as_frame=True,
        parser='auto',
    ).frame

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Id', 'SalePrice'], axis=1),
        data['SalePrice'], test_size=0.3, random_state=0)

We'll apply the selector to the numerical variables, so let's capture their names in a
list:

.. code:: python

    cols = [var for var in X_train.columns if X_train[var].dtypes !='O']

Now, we set up the wrapper with the SelectKBest, and fit it to the train set:

.. code:: python

    selector = SklearnTransformerWrapper(
        transformer = SelectKBest(f_regression, k=5),
        variables = cols)

    selector.fit(X_train.fillna(0), y_train)

Finally, we select the 5 best features in the train and test sets:

.. code:: python

    # transform the data
    X_train_t = selector.transform(X_train.fillna(0))
    X_test_t = selector.transform(X_test.fillna(0))

Even though feature-engine has its own implementation of OneHotEncoder, you may want
to use scikit-learn's transformer in order to access different options,
such as drop first category.
In the following example, we show you how to apply scikit-learn's OneHotEncoder to a
subset of categories using the :class:`SklearnTransformerWrapper()`. We start with the
imports and a function to load and clean the Titanic dataset:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from feature_engine.wrappers import SklearnTransformerWrapper

    # Load dataset
    def load_titanic():
        data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
        data = data.replace('?', np.nan)
        data['cabin'] = data['cabin'].astype(str).str[0]
        data['pclass'] = data['pclass'].astype('O')
        data['embarked'].fillna('C', inplace=True)
        data.drop(["name", "home.dest", "ticket", "boat", "body"], axis=1, inplace=True)
        return data

Next, we load the data and split it into a train set and a test set:

.. code:: python

    df = load_titanic()

    X_train, X_test, y_train, y_test= train_test_split(
        df.drop("survived", axis=1),
        df["survived"],
        test_size=0.2,
        random_state=42,
    )

Now, we set up the wrapper with the OneHotEncoder, fit it to the train set, and use it
to transform the train and test sets:

.. code:: python

    ohe = SklearnTransformerWrapper(
            OneHotEncoder(sparse=False, drop='first'),
            variables = ['pclass','sex'])

    ohe.fit(X_train)

    X_train_transformed = ohe.transform(X_train)
    X_test_transformed = ohe.transform(X_test)

We can examine the result by executing the following:

.. code:: python

   print(X_train_transformed.head())

The resulting dataframe is:

.. code:: python

         age  sibsp  parch     fare cabin embarked  pclass_2  pclass_3  sex_male
    772   17      0      0   7.8958     n        S       0.0       1.0       1.0
    543   36      0      0     10.5     n        S       1.0       0.0       1.0
    289   18      0      2    79.65     E        S       0.0       0.0       0.0
    10    47      1      0  227.525     C        C       0.0       0.0       1.0
    147  NaN      0      0     42.4     n        S       0.0       0.0       1.0


Let's say you want to use :class:`SklearnTransformerWrapper()` in a more complex
context. As you may note there are `?` signs to denote unknown values. Due to the
complexity of the transformations needed we'll use a Pipeline to impute missing values,
encode categorical features and create interactions for specific variables using
scikit-learn's PolynomialFeatures. We start with the imports:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from feature_engine.datasets import load_titanic
    from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
    from feature_engine.encoding import OrdinalEncoder
    from feature_engine.wrappers import SklearnTransformerWrapper

Next, we load the Titanic dataset and split it into a train set and a test set:

.. code:: python

    X, y = load_titanic(
        return_X_y_frame=True,
        predictors_only=True,
        cabin="letter_only",
    )

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

Now, we assemble the pipeline. The last step wraps scikit-learn's PolynomialFeatures with
:class:`SklearnTransformerWrapper()`, so that it only creates interactions between
`pclass` and `sex`:

.. code:: python

    pipeline = Pipeline(steps = [
        ('ci', CategoricalImputer(imputation_method='frequent')),
        ('mmi', MeanMedianImputer(imputation_method='mean')),
        ('od', OrdinalEncoder(encoding_method='arbitrary')),
        ('pl', SklearnTransformerWrapper(
            PolynomialFeatures(interaction_only = True, include_bias=False),
            variables=['pclass','sex']))
    ])

Finally, we fit the pipeline and use it to transform the train and test sets:

.. code:: python

    pipeline.fit(X_train)
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    print(X_train_transformed.head())

We see the resulting dataframe, with the new interaction features at the end:

.. code:: python

               age  sibsp  parch      fare  cabin  embarked  pclass  sex  \
    772  17.000000      0      0    7.8958      0         0     3.0  0.0
    543  36.000000      0      0   10.5000      0         0     2.0  0.0
    289  18.000000      0      2   79.6500      1         0     1.0  1.0
    10   47.000000      1      0  227.5250      2         1     1.0  0.0
    147  29.532738      0      0   42.4000      0         0     1.0  0.0

         pclass sex
    772         0.0
    543         0.0
    289         1.0
    10          0.0
    147         0.0


More details
^^^^^^^^^^^^

In the following Jupyter notebooks you can find more details about how to navigate the
parameters of the :class:`SklearnTransformerWrapper()` and also access the parameters
of the scikit-learn transformer wrapped, as well as the output of the transformations.

- `Wrap sklearn categorical encoder <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/wrappers/Sklearn-wrapper-plus-Categorical-Encoding.ipynb>`_
- `Wrap sklearn KBinsDiscretizer <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/wrappers/Sklearn-wrapper-plus-KBinsDiscretizer.ipynb>`_
- `Wrap sklearn SimpleImputer <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/wrappers/Sklearn-wrapper-plus-SimpleImputer.ipynb>`_
- `Wrap sklearn feature selectors <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/wrappers/Sklearn-wrapper-plus-feature-selection.ipynb>`_
- `Wrap sklearn scalers <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/wrappers/Sklearn-wrapper-plus-scalers.ipynb>`_

The notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
