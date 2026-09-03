import os
import pickle
from multiprocessing import cpu_count
from os import mkdir

import seaborn as sns
import time

import pandas
import pm4py

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path as Path

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from itertools import product
from typing import List, Optional, Any
import pandas as pd

import random

from approach import configurations as config

from statsmodels.tsa.stattools import grangercausalitytests


from matplotlib import rcParams

# Set the default font globally to Times New Roman
rcParams['font.family'] = 'Times New Roman'


def determine_independent_columns(causality_report, significance_level=0.05):
    """
    This function analyzes the Granger causality report to determine which columns are independent
    and which are dependent on other columns. Dependent columns are excluded from further analysis.

    Parameters:
    causality_report (pd.DataFrame): The output DataFrame from the Granger causality analysis containing p-values.
    significance_level (float): The threshold for Granger causality significance (default: 0.05).

    Returns:
    independent_columns (list): A list of columns that are independent (i.e., not Granger-caused by others).
    dependent_columns (list): A list of columns that are dependent (i.e., Granger-caused by others).
    """

    columns = causality_report.columns
    dependent_columns = set()

    # Loop through the causality report
    for col_y in columns:
        for col_x in columns:
            if col_x != col_y:
                # If the p-value is below the significance level, col_y is Granger-caused by col_x
                if causality_report.loc[col_y, col_x] < significance_level:
                    dependent_columns.add(col_y)

    # Independent columns are those not in the dependent set
    independent_columns = [col for col in columns if col not in dependent_columns]

    return independent_columns, list(dependent_columns)



def calc_and_plot_correlation_analysis(df, max_lag, path, file_name):


    # Get column names
    columns = df.columns

    # Initialize DataFrames to store maximum correlations and best lags
    correlation_report = pd.DataFrame(index=columns, columns=columns)
    best_lag_report = pd.DataFrame(index=columns, columns=columns)

    # Create a dictionary to store correlations for plotting curves
    correlation_dict = {}

    # Loop through each pair of time series
    for col_x in columns:
        for col_y in columns:
            if col_x != col_y:
                # Store correlations for different lags
                correlations = []

                # Calculate correlation for each lag
                for lag in range(0, max_lag + 1):
                    # Shift col_x by 'lag' positions
                    shifted_x = df[col_x].shift(lag)
                    corr = df[col_y].corr(shifted_x)
                    correlations.append(corr)

                # Store correlations for plotting
                correlation_dict[(col_x, col_y)] = correlations

                # Store the maximum correlation and the corresponding lag
                max_corr = max(correlations, key=abs)  # Max absolute correlation
                best_lag = correlations.index(max_corr)

                correlation_report.loc[col_y, col_x] = max_corr
                best_lag_report.loc[col_y, col_x] = best_lag

    # Convert reports to float type
    correlation_report = correlation_report.astype(float)
    best_lag_report = best_lag_report.astype(float)

    # Define the full path for the plot file
    output_directory = define_output_directory(os.path.dirname(path))
    file_path = os.path.join(output_directory, file_name + "_feature_correlation.png")

    # Check if the file exists and if it is older than 3 hours (10800 seconds)
    if os.path.exists(file_path):
        file_mod_time = os.path.getmtime(file_path)
        current_time = time.time()
        file_age = current_time - file_mod_time
        if file_age < 10800:  # 3 hours = 10800 seconds
            print(f"The file '{file_path}' is less than 3 hours old. No need to re-run the function.")
            return correlation_report, best_lag_report


    # Plotting: Subplots for each time series correlation to others at different lags
    num_series = len(columns)
    fig, axes = plt.subplots(nrows=num_series, ncols=1, figsize=(10, 4 * num_series))

    for i, col_x in enumerate(columns):
        ax = axes[i] if num_series > 1 else axes  # Handle case for 1 subplot
        for col_y in columns:
            if col_x != col_y:
                corr_vals = correlation_dict[(col_x, col_y)]
                ax.plot(range(0, max_lag + 1), corr_vals, label=f'{col_x} lags {col_y}')

        ax.axhline(y=0, color='r', linestyle='--', label='Zero correlation')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Lagged Correlation: {col_x} to others')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # Save or show plot based on configuration
    if config.SAVE_PLOTS:
        plt.savefig(file_path)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    return correlation_report, best_lag_report

def calc_and_plot_granger_analysis(df, max_lag, path, file_name):
    # Get column names
    columns = df.columns

    # Initialize DataFrames to store p-values and best lags
    causality_report = pd.DataFrame(index=columns, columns=columns)
    best_lag_report = pd.DataFrame(index=columns, columns=columns)

    # Create a dictionary to store p-values for plotting curves
    p_values_dict = {}

    # Loop through each pair of time series
    for col_x in columns:
        for col_y in columns:
            if col_x != col_y:
                # Perform Granger causality test
                test_result = grangercausalitytests(df[[col_x, col_y]], max_lag, verbose=False)

                # Store p-values for each lag and find the lag with the minimum p-value
                p_values = [test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag + 1)]
                min_p_value = min(p_values)
                best_lag = p_values.index(min_p_value) + 1  # Lags are 1-indexed

                # Store results
                causality_report.loc[col_y, col_x] = min_p_value
                best_lag_report.loc[col_y, col_x] = best_lag

                # Store p-values for plotting
                p_values_dict[(col_x, col_y)] = p_values

    # Convert reports to float type
    causality_report = causality_report.astype(float)
    best_lag_report = best_lag_report.astype(float)

    # Define the full path for the plot file
    output_directory = define_output_directory(os.path.dirname(path))
    file_path = os.path.join(output_directory, file_name + "_granger_causality.png")

    # Check if the file exists and if it is older than 3 hours (10800 seconds)
    if os.path.exists(file_path):
        file_mod_time = os.path.getmtime(file_path)
        current_time = time.time()
        file_age = current_time - file_mod_time
        if file_age < 10800:  # 3 hours = 10800 seconds
            print(f"The file '{file_path}' is less than 3 hours old. No need to re-run the function.")
            return causality_report, best_lag_report


    # Plotting: Subplots for each time series causing other series
    num_series = len(columns)
    fig, axes = plt.subplots(nrows=num_series, ncols=1, figsize=(10, 4 * num_series))

    for i, col_x in enumerate(columns):
        ax = axes[i] if num_series > 1 else axes  # Handle case for 1 subplot
        for col_y in columns:
            if col_x != col_y:
                p_vals = p_values_dict[(col_x, col_y)]
                ax.plot(range(1, max_lag + 1), p_vals, label=f'{col_x} causes {col_y}')

        ax.axhline(y=0.05, color='r', linestyle='--', label='Significance level (0.05)')
        ax.set_xlabel('Lag')
        ax.set_ylabel('p-value')
        ax.set_title(f'Granger Causality: {col_x} to others')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # Save or show plot based on configuration
    if config.SAVE_PLOTS:
        plt.savefig(file_path)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    return causality_report, best_lag_report


def plot_correlation_matrix(df, path_to_log, file_name):
    """
    This function takes a pandas DataFrame as input, calculates the correlation matrix,
    and displays a heatmap of the correlations.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to analyze.

    Returns:
    None: Displays the correlation matrix heatmap.
    """
    # Define the full path for the plot file
    output_directory = define_output_directory(os.path.dirname(path_to_log))
    file_path = os.path.join(output_directory, file_name + "_feature_correlation_matrix.png")

    # Check if the file exists and if it is older than 3 hours (10800 seconds)
    if os.path.exists(file_path):
        file_mod_time = os.path.getmtime(file_path)
        current_time = time.time()
        file_age = current_time - file_mod_time
        if file_age < 10800:  # 3 hours = 10800 seconds
            print(f"The file '{file_path}' is less than 3 hours old. No need to re-run the function.")
            return None

    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

    # Add a title
    plt.title('Correlation Matrix')

    # Save or show plot based on configuration
    if config.SAVE_PLOTS:
        plt.savefig(file_path)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    return None



def create_report(results):

    final_report_flattened = []
    for items in results:
        for item in items:
            (approach, args, window_size, log_path, process_feature, result) = item
            log_name = Path(log_path).name

            if result.empty:
                results_available = 'no'
            else:
                results_available = 'yes'
            row = {
                'log_name': log_name,
                'log_path': log_path,
                'process_feature': process_feature,
                'approach': approach,
                'window_size': window_size,
                'results_available': results_available
            }
            approach_parameters = {k: v for k, v in args.items() if k != "data"}
            entry = row | approach_parameters
            final_report_flattened.append(entry)

    final_report_flattened_df = pd.DataFrame(final_report_flattened)

    return final_report_flattened_df


def save_object_as_pkl(data, filename: str) -> None:
    """
    Save a list of tuples containing objects of different types to a file.

    Args:
        data: The list of tuples to be saved.
        filename (str): The name of the file to save the object to.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    return None


def load_interim_object(file_path: str) -> Any:
    """
    This function loads a pickled object from the interim directory.

    Args:
    scenario_name (str): The name of the scenario whose pickle file should be loaded.

    Returns:
    Any: The object loaded from the pickle file, or None if an error occurs.
    """

    # Open the pickle file and load its contents
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None



def extract_process_features_to_calc(yaml_path) -> list:
    """
    Load YAML file and extract the list for the 'window_steps' parameter.

    Args:
        yaml_path (Path): Path to the YAML file.

    Returns:
        list: List of values for 'window_steps'.
    """
    # Load the YAML content from the file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Extract the 'window_steps' list
    window_steps = data.get('process_features', [])

    return window_steps


def extract_window_steps(yaml_path) -> list:
    """
    Load YAML file and extract the list for the 'window_steps' parameter.

    Args:
        yaml_path (Path): Path to the YAML file.

    Returns:
        list: List of values for 'window_steps'.
    """
    # Load the YAML content from the file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Extract the 'window_steps' list
    window_steps = data.get('window_steps', [])

    return window_steps


def load_xes_files(directory: Path) -> List[Path]:
    """
    Load paths to all files that end with '.xes' in the given directory.

    Args:
        directory (str): The directory path to search for files.

    Returns:
        List[str]: A list of file paths ending with '.xes'.
    """
    # Use os.path.join to create full paths and filter files that end with '.xes'
    xes_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xes')]

    return xes_files



def load_ssd_parameters_new(scenario_folder, path_to_ssd_parameters: Optional[str] = config.DEFAULT_PARAMETER_DIR):

    # Load parameters for all ssd approaches
    with open(path_to_ssd_parameters, 'r') as stream:
        yml_configuration = yaml.safe_load(stream)


    paths_to_logs = load_xes_files(scenario_folder)
    window_sizes = yml_configuration['window_steps']
    function_and_args = {approach["function"]: (approach["params"])
             for approach in yml_configuration["approaches"].values()
             if approach.get("enabled", True)}

    log_path_and_window_combinations = list(product(paths_to_logs, window_sizes))


    # Initialize the list of tuples to store combinations
    funct_name_and_args_combinations = []

    for funcname, arguments in function_and_args.items():
        keys, values = zip(*arguments.items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        if config.TESTING:
            permutations_dicts = [random.choice(permutations_dicts)]
        funct_name_and_args_combinations += [(funcname, permutation) for permutation in permutations_dicts]

    combination_all = [(path, window, funcname, funcargs)
        for path, window in log_path_and_window_combinations
        for funcname, funcargs in funct_name_and_args_combinations]

    return combination_all

def load_ssd_parameters(process_feature_data,
                        path_to_ssd_parameters: Optional[str] = config.DEFAULT_PARAMETER_DIR):

    # Load parameters for all ssd approaches
    with open(path_to_ssd_parameters, 'r') as stream:
        parameters = yaml.safe_load(stream)

    parameters = {approach["function"]: (approach["params"])
             for approach in parameters["approaches"].values()
             if approach.get("enabled", True)}


    # Initialize the list of tuples to store combinations
    combinations = []

    for funcname, arguments in parameters.items():
        keys, values = zip(*arguments.items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        if config.TESTING:
            combinations += [(funcname, permutation) for permutation in [random.choice(permutations_dicts)]]
        else:
            combinations += [(funcname, permutation) for permutation in permutations_dicts]

    meta_args = [  # Arguments that all functions take.
        {
            "log_path": logpath,
            "window_size": window_size,
            "dataframe": dataframe,
            "log_df": log_df
        }
        for logpath, window_size, dataframe, log_df in process_feature_data
    ]

    arguments = [
        (funcname, arg_dict, meta_arg)
        for funcname, arg_dict in combinations
        for meta_arg in meta_args
    ]


    # Shuffle the Tasks
    np.random.shuffle(arguments)

    # Give each task an index for progress bar (only used if DO_SINGLE_BAR is False)
    arguments = [(funcname, d1, d2 | {"position": idx}) for idx, (funcname, d1, d2) in enumerate(arguments)]

    return arguments


def load_and_select_relevant_data(path, column_list):

    log_df = pm4py.read_xes(path)

    # Select columns that are in the list
    valid_columns = [col for col in column_list if col in log_df.columns]
    log_df_relevant = log_df[valid_columns]

    return log_df_relevant

def plot_process_features(df, path_to_log, file_name):
    # Define the full path for the plot file
    output_directory = define_output_directory(os.path.dirname(path_to_log))
    file_path = os.path.join(output_directory, file_name + '.png')

    # Check if the file exists and if it is older than 3 hours (10800 seconds)
    if os.path.exists(file_path):
        file_mod_time = os.path.getmtime(file_path)
        current_time = time.time()
        file_age = current_time - file_mod_time
        if file_age < 10800:  # 3 hours = 10800 seconds
            #print(f"The file '{file_path}' is less than 3 hours old. No need to re-run the function.")
            return None

    # Create subplots
    num_features = len(df.columns)
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 3.5 * num_features), sharex=True)


    mapping = {"case_completions": "Completed cases",
               "active_cases": "Active cases",
               "avg_lead_time": "Average lead time"}

    # Plot each feature in its respective subplot
    for ax, column in zip(axes, df.columns):
        ax.plot(df.index, df[column], marker='o', label=column)
        ax.set_title(mapping[column], size=28)
        #ax.set_ylabel('Values', size=20)
        ax.tick_params(axis='y', labelsize=24)  # Increase y-axis tick size

        ax.grid()
        #ax.legend()

    # Set overall title
    #fig.suptitle(f'{file_name}', fontsize=16)

    # Adjust layout to make room for the overall title
    fig.subplots_adjust(top=0.98)  # Increase top margin to avoid overlap

    # Set x-axis label
    plt.xlabel(r'Time windows ($W_l$)', size=28)
    plt.xticks(fontsize=24)

    # Adjust layout to fit everything nicely
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust rect to ensure tight layout respects the overall title space
    plt.show()

    # Plot process features
    # Save or show plot based on configuration
    if config.SAVE_PLOTS:
        plt.savefig(file_path)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()
    return None


def plot_process_features_in_one_figure(df):

    # Define the column name mapping
    mapping = {
        "case_completions": "Completed cases",
        "active_cases": "Active cases",
        "avg_lead_time": "Average lead time (in weeks)"
    }

    # Plot all features in one figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each feature on the same axes
    for column in df.columns:
        ax.plot(df.index, df[column], marker='o', label=mapping[column])

    # Set title and labels
    #ax.set_title('Combined Plot of Case Statistics', size=28)
    ax.set_xlabel(r'Time windows ($W_l$)', size=28)
    #ax.set_ylabel('Values', size=24)

    # Customize ticks
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    # Add a grid for better readability
    ax.grid()

    # Add a legend to distinguish the lines
    ax.legend(fontsize=20)

    # Adjust layout to fit everything nicely
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_process_feature_and_ssd_cognite(df, model, feature_name, window):
    # plot results for each feature, window step, and used SSD method
    df = df
    df['y_hat'] = model.y_hat
    df['y_hat'] = df.y_hat.shift(-1)
    df['p'] = model.p_steady
    df['p'] = df.p.shift(-1)
    plt.rcParams["figure.figsize"] = (20, 10)
    z = np.arange(0, len(df), 1)  # time scale
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Time (s)", fontsize='x-large')
    ax1.set_ylabel("Steady State Probability", color='black', fontsize='x-large')
    ax1.plot(df.index, df.p[0: len(z)], color='darkorange')  # noqa: E203
    ax1.tick_params(axis="y", labelcolor='black')
    ax1.grid(False)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel('Liquid Level (%)', color='black', fontsize='x-large')  # we already handled the x-label with ax1
    ax2.plot(df.index, df.iloc[:, 0].values, color='black')
    ax2.tick_params(axis="y", labelcolor='black')
    ax2.grid(False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Steady State Detection Over the Time Series ", fontsize='x-large')
    plt.savefig(f'fig_ssd_cognite_{feature_name}_{window}.png')
    plt.show()


def select_num_cores(num_cores: int = None):
    if num_cores is None:
        num_cores = cpu_count() - 2
    else:
        num_cores = min(num_cores, cpu_count() - 2)
    return num_cores


def aggregate_predictions(df, method='all'):
    """
    Aggregates binary predictions from a list of pd.Series.

    Parameters:
    predictions (list of pd.Series): List of pd.Series containing binary values (0 or 1).
    method (str, optional): Aggregation method to use. Options are:
                            - 'majority': Majority voting.
                            - 'average': Averaging.
                            - 'and': Logical AND.
                            - 'or': Logical OR.
                            - 'weighted_majority': Weighted majority voting.
                            - 'consensus': Consensus threshold voting.
                            - 'tie_breaker': Majority voting with a tie-breaker favoring 0.
                            - 'median': Median-based aggregation.
                            - 'custom_threshold': Majority voting with a custom threshold.
                            - 'max_voting': Max voting.
                            - 'min_voting': Min voting.
                            - 'mode': Mode aggregation.
                            - 'dempster': Dempster's combination rule.
                            - 'all': Return all aggregations as a dictionary.
                            Default is 'all'.
    weights (list of float, optional): Weights for each series in weighted majority voting.
    threshold (float, optional): Custom threshold for 'custom_threshold' and 'weighted_majority' methods.
    consensus_level (float, optional): Required consensus level for 'consensus' method (0-1).

    Returns:
    pd.Series or dict of pd.Series: Aggregated results.
    """

    results = {}

    # Majority Voting
    results['majority'] = (df.mean(axis=1) > 0.5).astype(int)
    # Logical AND
    results['and'] = df.all(axis=1).astype(int)

    # Logical OR
    results['or'] = df.any(axis=1).astype(int)

    for std_gaus_kernel in config.KERNELS:
        kernal_agg = agg_and_smooth_SSD_results(df, std_gaus_kernel=std_gaus_kernel)
        plot_SS_curve(kernal_agg, config.DEFAULT_OUTPUT_DIR, "fig_kernal_SS_curve.png")
        results[f'kernel_{std_gaus_kernel}'] = (kernal_agg > config.KERNEL_CONSENS).astype(int)

    results_df = pd.DataFrame(results)
    if method == 'all':
        return results_df
    else:
        return results_df[[method]]


def agg_and_smooth_SSD_results(df: pd.DataFrame, std_gaus_kernel: int = 4) -> pd.Series():
    """
       Aggregates a list of pd.Series by computing their mean and applies Gaussian smoothing.

       Parameters:
       - results_list (List[pd.Series]): A list of pd.Series objects with binary values (0 or 1).
       - std_gaus_kernel (int): Standard deviation for the Gaussian kernel used for smoothing.

       Returns:
       - pd.Series: A smoothed series with values normalized to the original mean range.
       """

    # Compute the mean across all series
    mean_series = df.mean(axis=1)

    # Apply Gaussian filter
    smoothed_series = gaussian_filter1d(mean_series, sigma=std_gaus_kernel)

    # Convert smoothed_series to a pandas Series
    smoothed_series = pd.Series(smoothed_series, index=mean_series.index)

    original_min = mean_series.min()
    original_max = mean_series.max()

    # Compute smoothed min and max
    smoothed_min = smoothed_series.min()
    smoothed_max = smoothed_series.max()

    # Handle edge case where smoothed range is zero
    if smoothed_max == smoothed_min:
        return mean_series  # Return original mean_series if no variation in smoothed_series

    # Normalize the smoothed data to match the original amplitude range
    normalized_smoothed_series = (smoothed_series - smoothed_min) / (smoothed_max - smoothed_min)
    normalized_smoothed_series = normalized_smoothed_series * (original_max - original_min) + original_min

    return normalized_smoothed_series


def plot_SS_curve(series: pd.Series, path_to_log, plot_name):
    """
    Plots a pandas Series.

    Parameters:
    series (pd.Series): The pandas Series to plot.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.

    Returns:
    None
    """

    if not isinstance(series, pd.Series):
        raise ValueError("The input must be a pandas Series.")

    plt.figure(figsize=(14, 7))
    plt.plot(series)

    # Title (optional if needed)
    #plt.title(plot_name, fontsize=30, pad=20)  # Set larger title font size and space it from the plot

    # Labels
    plt.xlabel(r'Time ($W_l$)', fontsize=28, labelpad=20)  # Increase x-axis label font size with padding
    plt.ylabel('Probability', fontsize=28, labelpad=20)  # Increase y-axis label font size with padding

    # Grid and tick parameters
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=24)  # Increase the font size of the ticks

    # Adjust the layout to avoid clipping of labels
    plt.tight_layout()


    # Save or show plot based on configuration
    if config.SAVE_PLOTS:
        plt.savefig(os.path.join(path_to_log, plot_name))
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()


def define_output_directory(path_to_logs):
    if config.DEFAULT_INPUT_DIR == path_to_logs:
        plot_path = config.DEFAULT_INTERIM_DIR
    else:
        plot_path = path_to_logs

    output_dir = os.path.join(plot_path, 'process_feature_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir





def plot_SS_curve_and_all_features(df: pd.DataFrame, agg_results: pd.DataFrame, path_to_log: str,
                                   plot_name: str) -> None:
    """
    Plots features from the given DataFrame alongside aggregated results and saves or displays the plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the features to plot. The index represents the window.
    - agg_results (pd.DataFrame): DataFrame containing aggregated results to plot on the secondary y-axis.
    - path_to_log (str): The path to the log file to define the output directory.
    - plot_name (str): The name of the plot to be saved or displayed.

    Returns:
    - None: This function does not return a value. It creates and saves/shows a plot.
    """



    # Set up the seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Determine the number of features
    num_features = len(df.columns)

    # Create subplots for each feature
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 4 * num_features), sharex=True)

    # Generate unique colors and line styles for each column in agg_results
    colors = sns.color_palette("husl", len(agg_results.columns))
    line_styles = ['-', '--', '-.', ':'] * (len(agg_results.columns) // 4 + 1)

    # Plot each feature
    for i, (ax, column) in enumerate(zip(axes, df.columns)):
        label = ' '.join(segment.capitalize() for segment in column.split('_'))
        ax.plot(df.index, df[column], marker='o', label=label)
        ax.set_title(column)
        ax.set_ylabel('Values')
        ax.grid()
        ax.legend(loc='upper left')

        # Create a secondary y-axis for agg_results
        ax2 = ax.twinx()
        for j, curve in enumerate(agg_results.columns):
            ax2.plot(
                agg_results.index, agg_results[curve],
                color=colors[j],
                linestyle=line_styles[j % len(line_styles)],
                label=f'Steady State {curve}'
            )
            ax2.fill_between(
                agg_results.index, agg_results[curve],
                color=colors[j], alpha=0.1
            )
        ax2.set_ylabel('Probability')
        ax2.legend(loc='upper right')

    # Set the overall plot title and adjust the layout
    fig.suptitle(plot_name, fontsize=16)
    fig.subplots_adjust(top=0.98)
    plt.xlabel('Window')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save or show plot based on configuration
    if config.SAVE_PLOTS:
        plt.savefig(os.path.join(define_output_directory(os.path.dirname(path_to_log)), plot_name + ".png"))
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    return None


def plot_process_features_and_SSD_results(
        df: pd.DataFrame,
        agg_results: pd.Series,
        file_name_input: tuple[str, dict[str, str]]
) -> None:
    """
    Plots features from the given DataFrame alongside aggregated results and saves or displays the plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the features to plot. The index represents the window.
    - agg_results (pd.Series): Series containing aggregated results to plot on the secondary y-axis.
    - file_name_input (tuple[str, dict[str, str]]): A tuple where the first element is a function name,
      and the second element is a dictionary with 'log_path' and 'window_size'.

    Returns:
    - None: This function does not return a value. It creates and saves/shows a plot.
    """

    func_name, meta_data = file_name_input
    log_name = os.path.splitext(os.path.basename(meta_data['log_path']))[0]
    plot_name = f"{log_name}_{meta_data['window_size']}_{func_name}.png"
    plot_title = f"{log_name} ({meta_data['window_size']}), SSD method: {func_name}"

    # Create subplots
    num_features = len(df.columns)
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 4 * num_features), sharex=True)

    # Plot each feature in its subplot
    for ax, column in zip(axes, df.columns):
        label = ' '.join(segment.capitalize() for segment in column.split('_'))
        ax.plot(df.index, df[column], marker='o', label=label)
        ax.set_title(column)
        ax.set_ylabel('Values')
        ax.grid()
        ax.legend(loc='upper left')

        # Create a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(agg_results.index, agg_results, color='orange', label='Steady State')
        ax2.fill_between(agg_results.index, agg_results, color='orange', alpha=0.5)
        ax2.set_ylabel('Probability')
        #ax2.set_ylabel('Probability', color='orange')
        #ax2.tick_params(axis='y', labelcolor='#FF8C00')
        ax2.legend(loc='upper right')

    # Set overall title and adjust layout
    fig.suptitle(plot_title, fontsize=16)
    fig.subplots_adjust(top=0.98)
    plt.xlabel('Window')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save or show plot based on configuration
    if config.SAVE_PLOTS:
        if config.DEFAULT_INPUT_DIR == os.path.dirname(meta_data['log_path']):
            plt.savefig(os.path.join(config.DEFAULT_OUTPUT_DIR, plot_name + ".png"))
        else:
            plt.savefig(os.path.join(os.path.dirname(meta_data['log_path']), plot_name + ".png"))
    if config.SAVE_PLOTS:
        plt.show()
    plt.close()
    return None


def create_string_from_arguments(approach_dict: dict) -> str:
    """
    Create a string from the values of the 'arguments' key in the given dictionary,
    with values joined by underscores.

    Parameters:
    approach_dict (dict): A dictionary containing an 'approach' and 'arguments' key.

    Returns:
    str: A string of argument values joined by underscores.
    """
    # Create a list of string representations of the argument values
    values_list = []
    for key, value in approach_dict.items():
        # Handle lists by joining their elements with an underscore
        if isinstance(value, list):
            values_list.extend([str(v) for v in value])
        else:
            values_list.append(str(value))

    # Join all values by underscores
    result_string = "_".join(values_list)

    return result_string


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} hours, {minutes} minutes, and {seconds} seconds"
