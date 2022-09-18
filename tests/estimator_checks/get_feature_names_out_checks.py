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
from sklearn.pipeline import Pipeline

from tests.estimator_checks.dataframe_for_checks import test_df


def check_get_feature_names_out(estimator):
    """
    Check that the method get_feature_names_out() returns the variable names of
    the transformed dataframe. That would be the same as the variable names in the
    train set used in fit(). The variable names are stored in `feature_names_in_`.

    For those transformers that return additional variables, we need to incorporate
    specific tests, based on the transformer functionality. They will be skipped from
    this test.
    """

    # train set
    X, y = test_df(categorical=True, datetime=True)

    # train transformer
    estimator = clone(estimator)

    # skip tests for transformers that add features
    if not hasattr(estimator, "_get_new_features_name"):
        estimator.fit(X, y)

        # train pipeline with transformer
        pipe = Pipeline([("transformer", clone(estimator))])
        pipe.fit(X, y)

        # feature names in train set
        feature_names = list(X.columns)

        # test transformer
        assert estimator.get_feature_names_out(input_features=None) == feature_names
        assert (
            estimator.get_feature_names_out(input_features=feature_names)
            == feature_names
        )

        # test transformer within pipeline
        assert pipe.get_feature_names_out(input_features=None) == feature_names
        assert pipe.get_feature_names_out(input_features=feature_names) == feature_names
