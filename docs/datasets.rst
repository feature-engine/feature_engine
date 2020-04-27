Datasets
========

The user guide and examples included in Feature-engine's documentation are based on these 3 datasets:

**Titanic dataset**

We use the dataset available in `openML <https://www.openml.org/d/40945>`_ which can be downloaded from `here <https://www.openml.org/data/get_csv/16826755/phpMYEkMl>`_.

**Ames House Prices dataset**

We use the data set created by Professor Dean De Cock:
* Dean De Cock (2011) Ames, Iowa: Alternative to the Boston Housing
* Data as an End of Semester Regression Project, Journal of Statistics Education, Vol.19, No. 3.

The examples are based on a copy of the dataset available on `Kaggle <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>`_.

The original data and documentation can be found here:

* `Documentation <http://jse.amstat.org/v19n3/decock/DataDocumentation.txt>`_

* `Data <http://jse.amstat.org/v19n3/decock/AmesHousing.xls>`_

**Credit Approval dataset**

We use the Credit Approval dataset from the UCI Machine Learning Repository:

Dua, D. and Graff, C. (2019). `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml>`_. Irvine, CA: University of California, School of Information and Computer Science.

To download the dataset visit this `website <http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/>`_ and click on "crx.data" to download the data set.

To prepare the data for the examples:

.. code:: python

    import random
    import pandas as pd
    import numpy as np

    # load data
    data = pd.read_csv('crx.data', header=None)

    # create variable names according to UCI Machine Learning information
    varnames = ['A'+str(s) for s in range(1,17)]
    data.columns = varnames

    # replace ? by np.nan
    data = data.replace('?', np.nan)

    # re-cast some variables to the correct types
    data['A2'] = data['A2'].astype('float')
    data['A14'] = data['A14'].astype('float')

    # encode target to binary
    data['A16'] = data['A16'].map({'+':1, '-':0})

    # save the data
    data.to_csv('creditApprovalUCI.csv', index=False)