"""From version 1.3 all transformers have the get_feature_names_out method, which
returns a list with the name of the variables in the transformed dataframe.

Some transformers can create and add new variables. Those need independent tests based
on their functionality. Many other transformers however, transform the variables in
place. For those transformers, get_feature_names_out either returns the entire list
of variables in the transformed dataframe, or the list of variables entered by the
user. The second is a bit useless, it is just included for compatibility with the
Scikit-learn Pipelne.
"""
from sklearn import clone

from tests.estimator_checks.dataframe_for_checks import test_df


def check_get_feature_names_out(estimator):
    """
    Check that the method get_feature_names_out() returns the variable names of
    the transformed dataframe. In most transformers that would be the same as
    the variable names in the train set used in fit(). The value is stored in
    `feature_names_in_`.

    For those transformers that return additional variables, we need to incorporate
    specific tests, based on the transformer functionality. They will be skipped from
    this test.
    """
    _skip_test = [
        "OneHotEncoder",
        "AddMissingIndicator",
        "LagFeatures",
        "WindowFeatures",
        "ExpandingWindowFeatures",
        "MathFeatures",
        "CyclicalFeatures",
        "RelativeFeatures",
        "DatetimeFeatures",
    ]
    # the estimator learns the parameters from the train set
    X, y = test_df(categorical=True, datetime=True)
    estimator = clone(estimator)
    estimator.fit(X, y)

    if estimator.__class__.__name__ not in _skip_test:
        # selection transformers
        if (
            hasattr(estimator, "confirm_variables")
            or estimator.__class__.__name__ == "DropFeatures"
        ):
            feature_names = [
                f for f in X.columns if f not in estimator.features_to_drop_
            ]
            assert estimator.get_feature_names_out() == feature_names
            assert estimator.transform(X).shape[1] == len(feature_names)

        else:
            assert estimator.get_feature_names_out() == [
                "var_" + str(i) for i in range(12)
            ] + [
                "cat_var1",
                "cat_var2",
                "date1",
                "date2",
            ]
