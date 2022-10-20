import pandas as pd
import numpy as np


def load_titanic(return_X_y_frame=False, predictors_only=False, raw=False):
    df = pd.read_csv("https://www.openml.org/data/get_csv/16826755/phpMYEkMl")

    if predictors_only:
        df.drop(
            columns=["name", "ticket", "home.dest", "boat", "body", "cabin"],
            inplace=True,
        )

    if not raw:
        df = df.replace("?", np.nan)

    if return_X_y_frame:
        X = df.drop(columns="survived")
        y = df["survived"]
        return X, y
    else:
        return df
