import numpy as np
import pandas as pd

from feature_engine.discretisation.binarizer import Binarizer

np.random.seed(42)
X = pd.DataFrame(dict(x = np.random.randint(1, 100, 100)))

b = Binarizer(threshold=200, variables=['x'])

b.fit(X)
