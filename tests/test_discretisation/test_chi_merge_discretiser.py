import pandas as pd
import pytest
from sklearn import datasets

from feature_engine.discretisation import ChiMergeDiscretiser

# TODO: Should we create the df here on in conftest?

# create dataset for unit tests
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_data = datasets.load_iris().data
iris = pd.DataFrame(iris_data, columns=col_names)
iris["flower"] = datasets.load_iris().target