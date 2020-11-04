import numpy as np

def get_feature_importances(estimator):


    """Retrieve feature importances from a fitted estimator"""

    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)

    if importances is None and coef_ is not None:

        importances = np.abs(coef_)

    return list(importances)
