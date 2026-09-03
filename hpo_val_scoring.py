"""Shared HPO-time validation scoring protocol for trace-level PPM models.

"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error

from create_prefixes_from_windows import make_three_way_split
from time_series_creation import (
    create_avg_throughtput_time_timeseries,
    create_concurrent_cases_timeseries,
)


def build_val_as_test_df(df: pd.DataFrame, train_split) -> pd.DataFrame:
    """HPO-time prediction input: cases observable at train_split.

    make_three_way_split with val_split=train_split repurposes the "test"
    partition to mean "cases active as of train_split" -- in-flight cases
    get their real prefix up to that point, fresh cases get their first
    event. True val/test data is never touched as model input.
    """
    _, _, val_as_test_df = make_three_way_split(
        df, case_col="caseid", time_col="end_timestamp",
        train_split=train_split, val_split=train_split, full_traces=True,
    )
    return val_as_test_df


def rem_time_to_event_log(rt_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a per-case remaining-time prediction table (amiri's rem_time.csv
    shape: caseid, start_timestamp, anchor_timestamp, rem_time_days) into the
    (caseid, end_timestamp) event log shape kpi_mae()/create_*_timeseries()
    expect -- one row for the case's start, one for its predicted end.


    """
    start_ts = pd.to_datetime(rt_df["start_timestamp"], utc=True, format="mixed")
    anchor = pd.to_datetime(rt_df["anchor_timestamp"], utc=True, format="mixed")
    end_ts = anchor + pd.to_timedelta(rt_df["rem_time_days"].astype(float), unit="D")
    start_ts = start_ts.dt.tz_convert(None)
    end_ts = end_ts.dt.tz_convert(None)
    return pd.concat([
        pd.DataFrame({"caseid": rt_df["caseid"].values, "end_timestamp": start_ts.values}),
        pd.DataFrame({"caseid": rt_df["caseid"].values, "end_timestamp": end_ts.values}),
    ], ignore_index=True)


def kpi_mae(event_log: pd.DataFrame, cc_val: pd.Series, tt_val: pd.Series):
    """Aggregate a predicted event log into CC/TT series over cc_val/tt_val's
    own date index and return (cc_mae, tt_mae) against those real actuals.


    """
    pred_cc = create_concurrent_cases_timeseries(
        event_log, time_col="end_timestamp", case_col="caseid", window="days", plot=False)
    pred_tt = create_avg_throughtput_time_timeseries(
        event_log, time_col="end_timestamp", case_col="caseid", window="days", plot=False)

    cc_arr = pred_cc.reindex(cc_val.index).ffill().bfill().fillna(0).to_numpy()
    tt_arr = pred_tt.reindex(tt_val.index).ffill().bfill().fillna(0).to_numpy()

    cc_mae = mean_absolute_error(cc_val.to_numpy(), cc_arr)
    tt_mae = mean_absolute_error(tt_val.to_numpy(), tt_arr)
    return cc_mae, tt_mae
