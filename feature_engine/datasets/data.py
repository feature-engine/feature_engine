import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer


def load_titanic(return_X_y_frame=False, predictors_only=False, handle_missing=False):
    df = pd.read_csv("https://www.openml.org/data/get_csv/16826755/phpMYEkMl")
    df = df.replace("?", np.nan)

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
