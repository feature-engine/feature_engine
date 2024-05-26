import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from feature_engine.imputation import CategoricalImputer, MeanMedianImputer


# TODO: loading the dataset from the internet is not the best, we need to store it
def load_titanic(
    return_X_y_frame=False, predictors_only=False, handle_missing=False, cabin=None
):
    """
    The load_titanic() function returns the well-known titanic dataset.

    Note that you need to have an internet connection for this function to work, as we
    are calling the dataset stored in `openML <https://www.openml.org/d/40945>`_ which
    can be downloaded from
    `here <https://www.openml.org/data/get_csv/16826755/phpMYEkMl>`_.

    Parameters
    ----------
    return_X_y_frame: bool, default=False
        If `True`, it returns a DataFrame (X) with the predictors and a Series (y) with
        the target variable. If `False`, it returns a single DataFrame with predictors
        and target.

    predictors_only: bool, default=False
        If `False`, it returns all the variables from the original Titanic Dataset. If
        `True`, it reurns only relevant predictors.

    handle_missing: bool, default=False
        If `False`, it returns the original dataset with missing values. If `True`,
        missing data is replaced with the string "Missing" in categorical variables and
        the mean in numerical variables.

    cabin: str, default=None
        If `None`, it returns the variable cabin as in the original data. If 'drop', it
        removes the variable from the data. If 'letter_only' it returns just the first
        letter of the cabin, without the number.

    Examples
    --------

    >>> from feature_engine.datasets import load_titanic
    >>> data = load_titanic(predictors_only=True, cabin="drop")
    >>> print(data.head())
       pclass  survived     sex      age  sibsp  parch      fare embarked
    0       1         1  female  29.0000      0      0  211.3375        S
    1       1         1    male   0.9167      1      2  151.5500        S
    2       1         0  female   2.0000      1      2  151.5500        S
    3       1         0    male  30.0000      1      2  151.5500        S
    4       1         0  female  25.0000      1      2  151.5500        S
    """
    # param checks
    if not isinstance(return_X_y_frame, bool):
        raise ValueError(
            "return_X_y_frame takes only booleans True and False. "
            f"Got {return_X_y_frame} instead."
        )

    if not isinstance(predictors_only, bool):
        raise ValueError(
            "predictors_only takes only booleans True and False. "
            f"Got {predictors_only} instead."
        )

    if not isinstance(handle_missing, bool):
        raise ValueError(
            "handle_missing takes only booleans True and False. "
            f"Got {handle_missing} instead."
        )

    if cabin is not None:
        if not isinstance(cabin, str) or cabin not in ["letter_only", "drop"]:
            raise ValueError(
                "the parameter 'cabin' takes only values None, 'letter_only' and "
                f"'drop'. Got {cabin} instead."
            )

    # load and prepare data
    df = pd.read_csv("https://www.openml.org/data/get_csv/16826755/phpMYEkMl")
    df = df.replace("?", np.nan)
    df["age"] = df["age"].astype("float64")
    df["fare"] = df["fare"].astype("float64")

    if predictors_only:
        df = df.drop(
            columns=["name", "ticket", "home.dest", "boat", "body"],
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

    if cabin == "letter_only":
        df["cabin"] = df["cabin"].astype(str).str[0]
    elif cabin == "drop":
        df = df.drop(columns=["cabin"])

    if return_X_y_frame:
        X = df.drop(columns="survived")
        y = df["survived"]
        return X, y
    else:
        return df
