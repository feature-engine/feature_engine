import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import check_X,check_X_y


class ForecastingPipeline(Pipeline):
    """
    TBD
    """

    def __init__(
            self,
            steps,
            forecasting_horizon: int = None,
            period: int = 1,
            freq: str = None,
            memory:bool=None,
            verbose:bool=False,
    ) -> None:

        if not isinstance(forecasting_horizon, int):
            raise ValueError(f"forecasting_horizon must be an integer. Got {forecasting_horizon} instead.")

        if not isinstance(period, int):
            raise ValueError(f"period must be an integer. Got {period} instead.")

        if not isinstance(freq, str) or not freq in ["Y", "M", "D", "h", "m", "s"]:
            raise ValueError(
                "freq takes strings 'Y', 'M','D', 'h', 'm', 's'. "
                f"Got {freq} instead."
            )

        self.steps = steps
        self.forecasting_horizon = forecasting_horizon
        self.period = period
        self.freq = freq
        self.memory = memory
        self.verbose = verbose

    def _get_steps(self):
        _, estimators = zip(*self.steps)
        return estimators

    def _horizon(self, X):

        forecast_start = X.index[-1] + np.timedelta64(self.period, self.freq)
        forecast_end = forecast_start + np.timedelta64(self.forecasting_horizon, self.freq)

        horizon = pd.date_range(
            start=forecast_start,
            end=forecast_end,
            freq=self.freq,
        )

        return horizon

    def fit(self, X, y):
        Xt, y = check_X_y(X, y)

        # Create features
        for _, _, transform in self._iter():
            if hasattr(transform, "transform"):
                Xt = transform.fit_transform(Xt)

            else:
                # Check that input feature was removed
                if any([f for f in X.columns if f in Xt.columns]):
                    raise ValueError(
                        "The predictive features contain the raw time series. "
                    )
                # Check if observations where removed after
                # creating features, and if so, adjust y.
                if len(Xt)!= len(y):
                    y.loc[Xt.index]

                # Fit regression model
                transform.fit(Xt, y)

        return self

    def transform(self, X):

        Xt = X.copy()
        for _, _, transform in self._iter():
            # need this tweak in case last class is not a
            # transformer
            if hasattr(transform, "transform"):
                Xt = transform.transform(Xt)
        return Xt

    def forecast(self, X):

        # check method fit has been called
        check_is_fitted(self)

        X = check_X(X)

        horizon = self._horizon(X)

        X = pd.concat([X, pd.DataFrame(index=horizon, columns=X.columns)], axis=0)

        for step in horizon:

            input_data = X[X.index < step]

            X.loc[step] = self.predict(input_data)[-1]

        return X.loc[horizon]

    def get_feature_names_out(self, input_features=None):
        if hasattr(self._get_steps()[-1], "transform"):
            return self._get_steps()[-1].get_feature_names_out(input_features)
        else:
            return self._get_steps()[-2].get_feature_names_out(input_features)
