"""
Compute the 'cases started per day' time series from an event log and
fit/predict with a Prophet model for arrival rate forecasting.

The arrival series is distinct from CC/TT: it counts how many NEW cases
begin (first event) on each calendar day, regardless of when they finish.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import numpy as np


def compute_arrival_series(
    df: pd.DataFrame,
    case_col: str = "caseid",
    time_col: str = "end_timestamp",
    freq: str = "D",
) -> pd.Series:
    """
    Return a daily 'cases started' Series with a DatetimeIndex.

    A case's start date is the minimum timestamp among all its events.
    Days with no arrivals are filled with 0.

    Parameters
    ----------
    df       : event log DataFrame
    case_col : name of the case-ID column
    time_col : name of the timestamp column
    freq     : pandas offset alias for bucketing; default 'D' (calendar day)
    """
    ts = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)
    case_starts = df.assign(_ts=ts).groupby(case_col)["_ts"].min()
    daily = (
        case_starts.dt.floor(freq)
        .value_counts()
        .sort_index()
        .rename("arrivals")
    )
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq=freq)
    return daily.reindex(full_idx, fill_value=0)


class ProphetArrivalModel:
    """
    Prophet model trained on the 'cases started per day' series.

    Fits on training data and predicts the number of new cases starting
    on each day of the test period.  Note: this is a SEPARATE Prophet
    model from the direct CC/TT baseline in prophet_baseline.py.
    """

    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self._model = None

    def fit(self, arrival_series: pd.Series) -> "ProphetArrivalModel":
        """
        Parameters
        ----------
        arrival_series : pd.Series
            DatetimeIndex (tz-naive) → int count of cases starting on that day.
            Missing days should already be filled with 0 (use compute_arrival_series).
        """
        from prophet import Prophet

        df = pd.DataFrame({
            "ds": arrival_series.index.tz_localize(None)
                  if arrival_series.index.tz is not None
                  else arrival_series.index,
            "y":  arrival_series.values.astype(float),
        })
        m = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df)
        self._model = m
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        """
        Predict daily arrivals for the given DatetimeIndex.

        Returns
        -------
        pd.Series with DatetimeIndex and predicted arrival counts (clipped ≥ 0).
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        future = pd.DataFrame({
            "ds": pd.DatetimeIndex(future_index).tz_localize(None)
                  if pd.DatetimeIndex(future_index).tz is not None
                  else pd.DatetimeIndex(future_index),
        })
        forecast = self._model.predict(future)
        result = pd.Series(
            forecast["yhat"].clip(lower=0).values,
            index=pd.DatetimeIndex(forecast["ds"]),
            name="predicted_arrivals",
        )
        return result
