import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer


def load_titanic(return_X_y_frame=False, predictors_only=False, handle_missing=False):
    """
    The load_titanic() function returns the well-known titanic dataset.

    Parameters
    ----------
    {variables}

    return_X_y_frame: bool, default=False
        If True returns separated X DataFrame for predictors and y Series for
        Target Variable.
        If False, returns a single DataFrame.

    predictors_only: bool, default=False
        If False return all the variables contained in the original Titanic Dataset.
        If True, only relevant predictors are kept dismissing the rest.

    handle_missing: bool, default=False
        If False, all missing values are shown as is. If True, proper imputations are
        applied: Missing Indicator for Categorical Values and Mean Imputation Numerical
        Variables.
    """
    df = pd.read_csv("https://www.openml.org/data/get_csv/16826755/phpMYEkMl")
    df = df.replace("?", np.nan)
    df["age"] = df["age"].astype("float64")
    df["fare"] = df["fare"].astype("float64")

    if predictors_only:
        df.drop(
            columns=["name", "ticket", "home.dest", "boat", "body", "cabin"],
            inplace=True,
        )

    if handle_missing:
        pipeline = Pipeline(
            steps=[
                (
                    "categorical_imputer",
                    CategoricalImputer(imputation_method="missing"),
                ),
                ("mean_median_imputer", MeanMedianImputer(imputation_method="mean")),
            ]
        )

        df = pipeline.fit_transform(df)

    if return_X_y_frame:
        X = df.drop(columns="survived")
        y = df["survived"]
        return X, y
    else:
        return df
