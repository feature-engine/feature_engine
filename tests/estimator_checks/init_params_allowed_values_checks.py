"""Many transformers have similar init parameters which take the same input values.
In this script, we add tests for the allowed values for those parameters.
"""
import pytest
from sklearn import clone


def check_error_param_missing_values(estimator):
    """
    Only for transformers with a parameter `missing_values`in init.

    Checks transformer raises error when user enters non-permitted value to the
    parameter.
    """
    # param takes values "raise" or "ignore"
    estimator = clone(estimator)
    for value in [2, "hola", False]:
        if estimator.__class__.__name__ == "MathFeatures":
            with pytest.raises(ValueError):
                estimator.__class__(
                    variables=["var_1", "var_2", "var_3"],
                    func="mean",
                    missing_values=value,
                )

        elif estimator.__class__.__name__ == "RelativeFeatures":
            with pytest.raises(ValueError):
                estimator.__class__(
                    variables=["var_1", "var_2", "var_3"],
                    reference=["var_4"],
                    func="mean",
                    missing_values=value,
                )
        else:
            with pytest.raises(ValueError):
                estimator.__class__(missing_values=value)
