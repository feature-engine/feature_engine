import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureElimination
