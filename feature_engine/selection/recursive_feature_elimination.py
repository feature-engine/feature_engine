import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine.selection.base_recursive_selector import BaseRecursiveSelector


class RecursiveFeatureElimination(BaseRecursiveSelector):
    """
    RecursiveFeatureElimination() selects features following a recursive elimination
    process.

    The process is as follows:

    1. Train an estimator using all the features.

    2. Rank the features according to their importance derived from the estimator.

    3. Remove the least important feature and fit a new estimator.

    4. Calculate the performance of the new estimator.

    5. Calculate the performance difference between the new and original estimator.

    6. If the performance drop is below the threshold the feature is removed.

    7. Repeat steps 3-6 until all features have been evaluated.

    Model training and performance evaluation are done with cross-validation.

    More details in the :ref:`User Guide <recursive_elimination>`.

    Parameters
    ----------
    estimator: object
        A Scikit-learn estimator for regression or classification.
        The estimator must have either a `feature_importances` or `coef_` attribute
        after fitting.

    variables: str or list, default=None
        The list of variable to be evaluated. If None, the transformer will evaluate
        all numerical features in the dataset.

    scoring: str, default='roc_auc'
        Desired metric to optimise the performance of the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float, int, default = 0.01
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold can be a big number.
        The threshold must be defined by the user. Bigger thresholds will select less
        features.

    cv: int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use cross_validate's default 5-fold cross validation

            - int, to specify the number of folds in a (Stratified)KFold,

            - CV splitter
                - (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)

            - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, KFold is used. These
        splitters are instantiated with `shuffle=False` so the splits will be the same
        across calls. For more details check Scikit-learn's `cross_validate`'s
        documentation.

    Attributes
    ----------
    initial_model_performance_ :
        Performance of the model trained using the original dataset.

    feature_importances_ :
        Pandas Series with the feature importance (comes from step 2)

    performance_drifts_:
        Dictionary with the performance drift per examined feature (comes from step 5).

    features_to_drop_:
        List with the features to remove from the dataset.

    variables_:
        The variables that will be considered for the feature selection.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the important features.
    transform:
         Reduce X to the selected features.
    fit_transform:
        Fit to data, then transform it.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features. Note that the selector trains various models at
        each round of selection, so it might take a while.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe
        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        X = super().fit(X, y)

        # Sort the feature importance values increasingly
        self.feature_importances_.sort_values(ascending=True, inplace=True)

        # to collect selected features
        _selected_features = []

        # temporary copy where we will remove features recursively
        X_tmp = X[self.variables_].copy()

        # we need to update the performance as we remove features
        baseline_model_performance = self.initial_model_performance_

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}

        # evaluate every feature, starting from the least important
        # remember that feature_importances_ is ordered already
        for feature in list(self.feature_importances_.index):

            # remove feature and train new model
            model_tmp = cross_validate(
                self.estimator,
                X_tmp.drop(columns=feature),
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=False,
            )

            # assign new model performance
            model_tmp_performance = model_tmp["test_score"].mean()

            # Calculate performance drift
            performance_drift = baseline_model_performance - model_tmp_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

            if performance_drift > self.threshold:

                _selected_features.append(feature)

            else:
                # remove feature and adjust initial performance
                X_tmp = X_tmp.drop(columns=feature)

                baseline_model = cross_validate(
                    self.estimator,
                    X_tmp,
                    y,
                    cv=self.cv,
                    return_estimator=False,
                    scoring=self.scoring,
                )

                # store initial model performance
                baseline_model_performance = baseline_model["test_score"].mean()

        self.features_to_drop_ = [
            f for f in self.variables_ if f not in _selected_features
        ]

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseRecursiveSelector.transform.__doc__
