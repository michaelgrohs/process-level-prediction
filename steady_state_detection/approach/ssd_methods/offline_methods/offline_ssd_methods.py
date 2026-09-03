from typing import Literal
import indsl
import numpy as np
import pandas as pd

def apply_ssd_using_rolling_windows(
    data: pd.Series,
    long_interval: int,
    short_interval: int,
    std_threshold: float,
    detect: Literal["decrease", "increase", "both"],
) -> pd.Series:
    """
    This function is based on the drift method indsl.detect.drift and was slightly adopted

    Drift.

    This function detects data drift (deviation) by comparing two rolling averages, short and long interval, of the signal. The
    deviation between the short and long term average is considered significant if it is above a given threshold
    multiplied by the rolling standard deviation of the long term average.

    Args:
        data: Time series.
        long_interval: Long length.
            Length of long term time interval.
        short_interval: Short length.
            Length of short term time interval.
        std_threshold: Threshold.
            Parameter that determines if the signal has changed significantly enough to be considered drift. The threshold
            is multiplied by the long term rolling standard deviation to take into account the recent condition of the
            signal.
        detect: Type.
            Parameter to determine if the model should detect significant decreases, increases or both. Options are:
            "decrease", "increase", or "both". Defaults to "both".

    Returns:
      pandas.Series: Boolean time series.
        Drift = 1, No drift = 0.
    """
    # Compute long term average and std
    long_average = data.rolling(long_interval).mean()
    long_stds = data.rolling(long_interval).std()

    # Compute short term average
    short_average = data.rolling(short_interval).mean()

    # Calculate drift masks
    drift_mask_increase = short_average > long_average + std_threshold * long_stds
    drift_mask_decrease = short_average < long_average - std_threshold * long_stds

    # set mask according to setting
    if detect == "increase":
        drift_mask = drift_mask_increase
    elif detect == "decrease":
        drift_mask = drift_mask_decrease
    else:
        drift_mask = np.logical_or(drift_mask_increase, drift_mask_decrease)

    # Focus on steady state periods: revers the values.
    drift_mask_revers = -drift_mask

    return drift_mask_revers.astype(int)


def apply_ssd_cumsum(data: pd.Series,
                     threshold,
                     drift,
                     detect,
                     predict_ending,
                     alpha,
                     return_series_type
                     ) -> pd.Series:
    '''
    Steady State detector: cumulative sum
    Documentations: check out indsl.detect.cusum() method

    References:
        https://nbviewer.org/github/demotu/detecta/blob/master/docs/detect_cusum.ipynb
    :return:
    '''
    result = indsl.detect.cusum(data=data,
                                threshold=threshold,
                                drift=drift,
                                detect=detect,
                                predict_ending=predict_ending,
                                alpha=alpha,
                                return_series_type=return_series_type)
    return (1 - result).astype(int)


def apply_ssd_ed_pelt(data: pd.Series, min_distance) -> pd.Series:
    '''
    Steady State detector: based on the ED Pelt change point detection algorithm
    Documentations: check out indsl.detect.cpd_ed_pelt() method

    :return
        pandas.Series: Binary time series.
        Steady state = 1, Transient = 0.
    '''
    result = indsl.detect.cpd_ed_pelt(data,min_distance=min_distance)

    return result.astype(int)


def apply_ssd_ed_pelt_with_transitions(data: pd.Series,
                                       min_distance: int,
                                       var_threshold: float,
                                       slope_threshold: float) -> pd.Series:
    '''
    Steady State detector: based on the ED Pelt change point detection algorithm and transition periods
    Documentations: check out indsl.detect.ssd_cpd() method

    :return
        pandas.Series: Binary time series.
        Steady state = 1, Transient = 0.
    '''
    result = indsl.detect.ssd_cpd(data.astype(float),
                                min_distance=min_distance,
                                var_threshold=var_threshold,
                                slope_threshold=slope_threshold)
    return result.astype(int)


def apply_ssd_by_rhinehart_2013(data: pd.Series,
                                ratio_lim: float,
                                alpha1: float,
                                alpha2: float,
                                alpha3: float) -> pd.Series:
    '''
    Steady State detector: variance filter
    For documentations check out: https://indsl.docs.cognite.com/detect.html (indsl.detect.ssid)

    References:
    Rhinehart, R. Russell. (2013).
    Automated steady and transient state identification in noisy processes.
    Proceedings of the American Control Conference. 4477-4493. 10.1109/ACC.2013.6580530

    :return:
        pandas.Series: Binary time series.
        Steady state = 1, transient = 0.
    '''
    result = indsl.detect.ssid(data, ratio_lim, alpha1, alpha2, alpha3)
    result = (1 - result).astype(int)

    # Make sure the original periods are not excluded and the output series has the same length
    combined_index = data.index.union(result.index)
    extended_series = result.reindex(combined_index, fill_value=0)

    return extended_series



