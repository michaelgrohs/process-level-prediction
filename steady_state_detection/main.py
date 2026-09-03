import os.path
import random
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
import approach.utilities as util
import approach.configurations as config
import approach.process_features as pf
from approach.ssd_methods.offline_methods.offline_ssd_methods import *
from multiprocessing import Pool, RLock, freeze_support
from typing import Dict
from itertools import product
import warnings
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import matplotlib.pyplot as plt
# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_steady_state_traces(
    log_df: pd.DataFrame,
    result: pd.DataFrame,
    threshold: float
) -> Dict[str, int]:
    """
    Analyzes cases in the log DataFrame to determine whether they meet
    a steady-state condition based on a threshold value.

    Args:
        log_df (pd.DataFrame): DataFrame containing event logs, where each event is associated with a case.
        result (pd.DataFrame): DataFrame containing the result values for analysis.
        threshold (float): The threshold to determine if a case is in steady state.

    Returns:
        Dict[str, int]: A dictionary mapping case names to 1 if they meet the steady-state condition,
                        otherwise 0.
    """
    steady_state_cases = {}

    # Group by case
    for case_name, case_data in log_df.groupby(config.ATTR_CASE):
        # Extract result values for the current case, adjusting for 0-based index
        case_SS_values = result.iloc[case_data[config.ATTR_WINDOW] - 1]

        # Calculate the mean value
        steady_state_percent = case_SS_values.mean()

        # Check if the mean exceeds the threshold
        steady_state_cases[case_name] = int(steady_state_percent > threshold)

    # Reverse the dictionary
    steady_state_periods = {}
    for key, value in steady_state_cases.items():
        steady_state_periods.setdefault(value, []).append(key)


    return steady_state_cases, steady_state_periods


def load_approach_arguments(log_paths: List[str]) -> List[Tuple[str, str, str, Dict[str, float]]]:
    """
    Generates a list of argument tuples for processing logs using specified SSD approaches.

    Args:
        log_paths (List[str]): A list of paths to log files.

    Returns:
        List[Tuple[str, str, str, Dict[str, float]]]: A list of tuples, each containing:
                                                      - A path to a log file (str).
                                                      - A window size attribute (str).
                                                      - The approach name (str).
                                                      - A dictionary of approach arguments with their respective values (Dict[str, float]).
    """
    # Prepare a list to store all combinations of log paths, window sizes, and SSD approach arguments
    all_combinations = []

    # Iterate over each SSD approach defined in the configuration
    for ssd in config.SSD_APPROACH:
        approach = ssd['approach']
        arguments = ssd['arguments']

        # Generate all possible combinations of argument values for the current SSD approach
        argument_combinations = list(product(*arguments.values()))

        # Generate combinations that include log paths, window size, the SSD approach, and argument combinations
        approach_combinations = list(product(log_paths, config.WINDOW_SIZE, [approach], argument_combinations))

        # Format each combination as a tuple and create a dictionary from the argument keys and values
        formatted_combinations = [
            (log_path, window_size, approach, dict(zip(arguments.keys(), argument_values)))
            for log_path, window_size, approach, argument_values in approach_combinations
        ]

        # Add the formatted combinations to the list of all combinations
        if config.TESTING:
            all_combinations = [random.choice(formatted_combinations)]
        else:
            all_combinations.extend(formatted_combinations)



    return all_combinations


def multiprocessing(input_folder=config.DEFAULT_INPUT_DIR,
                    num_cores: int = config.NUM_CORES,
                    approach_parameters: List[Tuple[str, str, str, Dict[str, float]]] = None):

    # Load paths to event logs in the input folder
    paths_to_logs = util.load_xes_files(input_folder)

    # Load configurations
    if approach_parameters is None:
        approach_parameters = load_approach_arguments(paths_to_logs)

    # Run multiprocessing
    num_cores = util.select_num_cores(num_cores)
    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention
    with Pool(num_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        results = list(
            tqdm(p.imap(core_function, approach_parameters), desc="Completed", total=len(approach_parameters)))

    return results


def load_or_calculate_process_features(path, window_step, file_name_and_window):
    path_to_pkl_pf = os.path.join(config.DEFAULT_INTERIM_DIR, 'pkl_process_features')
    path_to_pkl_log = os.path.join(config.DEFAULT_INTERIM_DIR, 'pkl_event_log')
    os.makedirs(path_to_pkl_pf, exist_ok=True)
    os.makedirs(path_to_pkl_log, exist_ok=True)
    pf_df = util.load_interim_object(os.path.join(path_to_pkl_pf, file_name_and_window + '_pf.pkl'))
    log_df_rel = util.load_interim_object(os.path.join(path_to_pkl_log, file_name_and_window + '_log.pkl'))
    if pf_df is None or log_df_rel is None:
        log_df = util.load_and_select_relevant_data(path, config.ATTRIBUTES)
        pf_df, log_df_rel = pf.derive_process_feature(log_df, window_step)
        util.save_object_as_pkl(pf_df, os.path.join(path_to_pkl_pf, file_name_and_window + '_pf.pkl'))
        util.save_object_as_pkl(log_df_rel, os.path.join(path_to_pkl_log, file_name_and_window + '_log.pkl'))

    return pf_df, log_df_rel

def print_ss_trace_statistics(file_name: str, steady_state_periods: dict) -> tuple[int, int, float]:
    steady_state_periods = steady_state_periods or {}

    ss_cases = steady_state_periods.get(1, [])
    nss_cases = steady_state_periods.get(0, [])

    n_SS_traces = len(ss_cases)
    n_total_traces = n_SS_traces + len(nss_cases)

    if n_total_traces > 0:
        ratio = round(n_SS_traces / n_total_traces, 4) * 100
    else:
        ratio = 0.0

    print(f"{file_name}: N. of SS traces: {n_SS_traces} ({ratio}%) ")

    return n_SS_traces, n_total_traces, ratio

def core_function(arg):

    # Prepare meta-variables
    path, window_step, funcname, funcargs = arg
    approach_parameters = {k: v for k, v in funcargs.items()}
    arg_str = util.create_string_from_arguments(approach_parameters)
    file_name = f"{os.path.splitext(os.path.basename(path))[0]}"

    folder_id = os.path.normpath(path).split(os.sep)[-2]
    file_name_and_window = folder_id + "_" + file_name + "_"  + str(window_step)

    # Load or calculate process features
    pf_df, log_df_rel = load_or_calculate_process_features(path, window_step, file_name_and_window)

    # TODO: remove the lines below
    # Plots
    #util.plot_process_features(pf_df, path, file_name_and_window)
    #util.plot_correlation_matrix(process_features_df, path, file_name_and_window)

    # Correlation and causality analysis
    #correlation_report, correlation_best_lag_report = util.calc_and_plot_correlation_analysis(log_df_rel, 10, path, file_name_and_window)
    #causality_report, causality_best_lag_report = util.calc_and_plot_granger_analysis(process_features_df, 10, path, file_name_and_window)

    # TODO: implement a decision logic for process feature independency selection
    #independent, dependent = util.determine_independent_columns(causality_report, significance_level=0.05)

    report_all = []
    SSD_results_per_feature = pd.DataFrame()
    for feature in pf_df.columns:
        funcargs['data'] = pf_df[feature]
        try:
            ss_result = globals()[funcname](**funcargs)
            ss_traces, steady_state_periods = get_steady_state_traces(log_df_rel, ss_result, config.SS_TRACE_THRESHOLD)
            #util.plot_SS_curve_and_all_features(pf_df, ss_result, path, f"testing____ss_aggregations")

            report = {
                "path": path,
                "window_step": window_step,
                "funcname": funcname,
                'ssd_agg_option': "no",
                "ssd_feature": feature,
                "ss_result": ss_result,  # Convert to list for storage
                "ss_traces": ss_traces,  # Convert to list for storage
                "ss_traces_per_period": steady_state_periods

            }
            entry = report | approach_parameters
            report_all.append(entry)

            # Keep track of all obtained ss curves to aggregate them later
            SSD_results_per_feature[feature] = ss_result
        except:
            print(f"Error: {funcname}, {arg_str}")

    # Aggregate and plot results
    SSD_confidence_curve = util.aggregate_predictions(SSD_results_per_feature, method=config.SSD_AGGREGATION)

    for ssd_agg_option in SSD_confidence_curve.columns:
        ss_curve_option = SSD_confidence_curve[[ssd_agg_option]]
        #util.plot_SS_curve_and_all_features(pf_df, ss_curve_option, path, file_name_and_window + f"_{ssd_agg_option}_ss_aggregations")

        # Consider simple smoothing
        #util.plot_SS_curve_and_all_features(SSD_results_per_feature, ss_curve_option, path, file_name_conf + f"_ss_aggregations_{ssd_agg_option}")
        #plot_SS_curve(kernal_agg, config.DEFAULT_OUTPUT_DIR, file_name + f"_plot_kernel_{std_gaus_kernel}.png")

        # Detect SS traces
        # TODO: make sure we remove try / except...
        try:
            ss_traces, steady_state_periods = get_steady_state_traces(log_df_rel, ss_curve_option, config.SS_TRACE_THRESHOLD)
            report = {
                "path": path,
                "window_step": window_step,
                "funcname": funcname,
                'ssd_agg_option': ssd_agg_option,
                "ssd_feature": "agg_feature",
                "ss_result": SSD_confidence_curve[ssd_agg_option],  # Convert to list for storage
                "ss_traces": ss_traces,  # Convert to list for storage
                "ss_traces_per_period": steady_state_periods
            }
        except:
            print(f"Error: {funcname}, {arg_str}, {path}")

        entry = report | approach_parameters
        report_all.append(entry)

    if not config.EVALUATION:
        print_ss_trace_statistics(file_name, steady_state_periods)

        # Save SS and NSS dictionary with case ids
        util.save_object_as_pkl(steady_state_periods, os.path.join(config.DEFAULT_OUTPUT_DIR, file_name + '_SS_&_NSS' + '.pkl'))

        # Save event log with SputtyS case
        log_df_rel_SS = log_df_rel[log_df_rel['case:concept:name'].isin(steady_state_periods[1])]
        xes_exporter.apply(log_df_rel_SS, os.path.join(config.DEFAULT_OUTPUT_DIR, file_name + "_SS.xes"))

    return report_all



if __name__ == "__main__":

    # Timing: start
    start_time = time.time()  # Record the start time

    # Run SSD (multiprocessing for all logs)
    results = multiprocessing()
    # Save or process the DataFrame as needed
    # TODO: create a proper report function tha converts things from a list of rows into a proper csv table

    #results.to_csv(os.path.join(config.DEFAULT_OUTPUT_DIR, 'SSD_report.csv'), index=False)


    #results, report = SSD_approach()
    #util.save_interim_object(results, os.path.join(config.DEFAULT_INTERIM_DIR, 'testing_results.pkl'))
    #report.to_csv(os.path.join(config.DEFAULT_OUTPUT_DIR, "SSD_detection_results.csv"), index=False)

    # Record the end time, calculate the duration, and print the statement
    end_time = time.time()
    execution_duration = end_time - start_time
    print(f"Execution duration: {execution_duration:.2f} seconds")

    # Close the program
    sys.exit()
