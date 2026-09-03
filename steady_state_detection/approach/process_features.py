import os.path

import numpy as np
import pandas as pd
from approach import configurations as config, utilities as util
from pathlib import Path as Path
from multiprocessing import Pool, RLock, freeze_support, cpu_count
import pm4py
from tqdm import tqdm




def process_features_calculation(path, window_step, file_name):

    log_df_rel = util.load_and_select_relevant_data(path, config.ATTRIBUTES)
    # Calculate process features
    process_features_df, log_df_rel = derive_process_feature(log_df_rel, window_step)
    # Plot process features
    util.plot_process_features(process_features_df, file_name)

    if True:
        util.save_object_as_pkl(process_features_df, os.path.join(config.DEFAULT_INTERIM_DIR, file_name + '.pkl'))

    return process_features_df, log_df_rel


def calculate_features_for_log_and_window(args):
    path, window_step = args

    file_name = f"{Path(path).name.split('.')[0]}_{window_step}"

    if config.LOAD_PROCESS_FEATURES:
        try:
            process_features_df = util.load_interim_object(os.path.join(config.DEFAULT_INTERIM_DIR, file_name))
        except:
            process_features_df, log_df = process_features_calculation(path, window_step, file_name)
    else:
        process_features_df, log_df = process_features_calculation(path, window_step, file_name)

    return (path, window_step, process_features_df, log_df)


def derive_process_feature(df, window_step):
    """
    Derive process features from the given DataFrame based on the specified window step.

    Parameters:
    - df: DataFrame containing event logs.
    - window_step: Time window step for feature calculation.

    Returns:
    - DataFrame with process features.
    """

    # Derive timeframe
    timeframe = get_timeframe(df, window_step)

    # Assign window ID to each event
    df_wip = df.copy()
    df_wip[config.ATTR_WINDOW] = df[config.ATTR_TIME].apply(lambda x: get_period(x, timeframe))

    # Get process features to calculate
    pf_scope = util.extract_process_features_to_calc(config.DEFAULT_PARAMETER_DIR)

    features = []

    # Calculate features based on scope
    if pf_scope.get('arrivals'):
        features.append(calc_case_arrival_and_completion_rates(df_wip, timeframe, 'arrival'))

    if pf_scope.get('completions'):
        features.append(calc_case_arrival_and_completion_rates(df_wip, timeframe, 'completion'))

    if pf_scope.get('active_cases'):
        features.append(calc_active_cases(df_wip, timeframe))

    if pf_scope.get('avg_lead_time'):
        features.append(calc_avg_lead_time(df_wip, timeframe))

    if pf_scope.get('executed_events'):
        features.append(calc_executed_events(df_wip, timeframe))

    if pf_scope.get('available_resources') and config.ATTR_RESOURCE in df_wip.columns.to_list():
        features.append(calc_available_resources_events(df_wip, timeframe))

    # Combine calculated process features
    if features:
        features = [features[0]] + [df.drop('window', axis=1) for df in features[1:]]
        process_features = pd.concat(features, axis=1, join='outer')
    else:
        # Handle case where no features are computed
        return pd.DataFrame()

    # Set index and update its type
    process_features.set_index(config.ATTR_WINDOW, inplace=True)
    process_features.index = pd.to_datetime(timeframe[1:])

    return process_features, df_wip


def calc_active_cases(df, timeframe):
    attr = 'active_cases'
    # Create a DataFrame for timeframe windows
    timeframe_windows = pd.DataFrame(range(1, len(timeframe)), columns=[config.ATTR_WINDOW])

    # Calculate active cases per window
    active_cases = df.groupby(config.ATTR_WINDOW)[config.ATTR_CASE].nunique().reset_index(name=attr)

    # Merge timeframe windows with attr values
    result = timeframe_windows.merge(
        active_cases,
        on='window',
        how='left'
    )
    result[attr] = result[attr].fillna(0)
    return result


def calc_executed_events(df, timeframe):
    attr = 'executed_events'
    # Create a DataFrame for timeframe windows
    timeframe_windows = pd.DataFrame(range(1, len(timeframe)), columns=[config.ATTR_WINDOW])

    # Calculate active cases per window
    active_cases = df.groupby(config.ATTR_WINDOW)[config.ATTR_CASE].count().reset_index(name=attr)

    # Merge timeframe windows with attr values
    result = timeframe_windows.merge(
        active_cases,
        on='window',
        how='left'
    )
    result[attr] = result[attr].fillna(0)
    return result


def calc_available_resources_events(df, timeframe):
    attr = 'available_resource'
    # Create a DataFrame for timeframe windows
    timeframe_windows = pd.DataFrame(range(1, len(timeframe)), columns=[config.ATTR_WINDOW])

    # Calculate active cases per window
    active_cases = df.groupby(config.ATTR_WINDOW)[config.ATTR_RESOURCE].nunique().reset_index(name=attr)

    # Merge timeframe windows with attr values
    result = timeframe_windows.merge(
        active_cases,
        on='window',
        how='left'
    )
    result[attr] = result[attr].fillna(0)
    return result


def get_timeframe(df, window_step):
    start_moment = min(df[config.ATTR_TIME])
    end_moment = max(df[config.ATTR_TIME])
    timeframe = pd.date_range(start=start_moment, end=end_moment, freq=window_step, normalize=True)
    # Extend the timeframe s.t. the first and last events are part of the first and last window.
    timeframe = timeframe.union(pd.date_range(end=timeframe[0], periods=2, freq=window_step))
    timeframe = timeframe.union(pd.date_range(start=timeframe[-1], periods=2, freq=window_step))
    return timeframe


def calc_avg_lead_time(df, timeframe):
    """Calculate the average lead time per window from the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing case data.
        timeframe (list): The list of time windows.

    Returns:
        pd.DataFrame: A DataFrame with average lead time for each window.
    """

    # Calculate case duration for each case
    case_duration = df.groupby(config.ATTR_CASE)[config.ATTR_TIME].agg(lambda x: x.max() - x.min())

    # Get the maximum end window for each case
    case_end_window = df.groupby(config.ATTR_CASE)[config.ATTR_WINDOW].max()

    # Merge case duration and end window into a new DataFrame
    merged_data = pd.DataFrame({'case_duration': case_duration, 'case_end_window': case_end_window})

    # Calculate the average lead time per window
    avg_lead_time = merged_data.groupby('case_end_window')['case_duration'].mean()

    # Create a DataFrame for timeframe windows
    timeframe_windows = pd.DataFrame(range(1, len(timeframe)), columns=[config.ATTR_WINDOW])

    # Merge average lead time with timeframe windows
    avg_lead_time = timeframe_windows.merge(
        avg_lead_time.rename('avg_lead_time'),
        left_on='window',
        right_on='case_end_window',
        how='left'
    )

    # Fill NaN values with 0 and convert to timedelta
    avg_lead_time['avg_lead_time'] = avg_lead_time['avg_lead_time'].fillna(pd.Timedelta(0))

    # Convert time units into seconds (float)
    window_step_in_sec = (timeframe[1] - timeframe[0]).total_seconds()
    avg_lead_time['avg_lead_time'] = avg_lead_time['avg_lead_time'].apply(lambda x: x.total_seconds()) / window_step_in_sec

    return avg_lead_time


def calc_case_arrival_and_completion_rates(df, timeframe, option):

    # Step 1: Find the earliest or the latest event for each case
    if option == 'arrival':
        attr = 'case_arrivals'
        earliest_timestamps = df.loc[df.groupby(config.ATTR_CASE)[config.ATTR_TIME].idxmin()]
    elif option == 'completion':
        attr = 'case_completions'
        earliest_timestamps = df.loc[df.groupby(config.ATTR_CASE)[config.ATTR_TIME].idxmax()]

    # Step 2: Select relevant columns
    earliest_timestamps = earliest_timestamps[[config.ATTR_CASE, config.ATTR_WINDOW]]

    # Step 3: Count unique concept:name per window
    result = earliest_timestamps.groupby(config.ATTR_WINDOW)[config.ATTR_CASE].count().reset_index()

    # Rename columns for clarity
    result.columns = [config.ATTR_WINDOW, attr]

    # Create a DataFrame for timeframe windows
    timeframe_windows = pd.DataFrame(range(1, len(timeframe)), columns=[config.ATTR_WINDOW])

    # Merge average lead time with timeframe windows
    result = timeframe_windows.merge(
        result,
        on=config.ATTR_WINDOW,
        how='left'
    )
    result[attr] = result[attr].fillna(0)

    return result


# Function to determine the period for each timestamp
def get_period(timestamp, timeframe):
    for i, (start, end) in enumerate(zip(timeframe[:-1], timeframe[1:])):
        if start <= timestamp < end:
            return i+1
    return np.nan