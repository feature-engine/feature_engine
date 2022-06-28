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

    if estimator.__class__.__name__ not in _skip_test:

        # train set
        X, y = test_df(categorical=True, datetime=True)

        # train transformer
        estimator = clone(estimator)
        estimator.fit(X, y)

        # train pipeline with transformer
        pipe = Pipeline([("transformer", clone(estimator))])
        pipe.fit(X, y)

        # feature names in train set
        feature_names = list(X.columns)

        # selection transformers
        if (
            hasattr(estimator, "confirm_variables")
            or estimator.__class__.__name__ == "DropFeatures"
        ):
            feature_names = [
                f for f in feature_names if f not in estimator.features_to_drop_
            ]

            # take a few as input features (selectors ignore this parameter)
            input_features = [feature_names[0:3]]

            # test transformer
            assert estimator.get_feature_names_out() == feature_names
            assert estimator.get_feature_names_out(input_features) == feature_names
            assert estimator.transform(X).shape[1] == len(feature_names)

            # test transformer within pipeline
            assert pipe.get_feature_names_out() == feature_names
            assert pipe.get_feature_names_out(input_features) == feature_names

        elif estimator.__class__.__name__ == "MatchVariables":
            # take a few as input features (these transformers ignore this parameter)
            input_features = [feature_names[0:3]]

            # test transformer
            assert estimator.get_feature_names_out() == feature_names
            assert estimator.get_feature_names_out(input_features) == feature_names
            assert estimator.transform(X).shape[1] == len(feature_names)

            # test transformer within pipeline
            assert pipe.get_feature_names_out() == feature_names
            assert pipe.get_feature_names_out(input_features) == feature_names

        else:
            input_features = estimator.variables_

            # test transformer
            assert estimator.get_feature_names_out() == feature_names
            assert estimator.get_feature_names_out(input_features) == input_features
            assert estimator.transform(X).shape[1] == len(feature_names)

            # test transformer within pipeline
            assert pipe.get_feature_names_out() == feature_names
            assert pipe.get_feature_names_out(input_features) == input_features
