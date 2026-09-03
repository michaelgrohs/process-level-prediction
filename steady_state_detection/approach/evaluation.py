import os

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score

from approach import utilities as util



def get_actual_steady_states(ss_detected, path) -> pd.Series:
    '''
    This function extracts the actual steady states from pkl file and can be used within the main function
    (in contrast to the function get_actual_steady_state_index from the script experiment_1).

    :param experiment:
    :return:
    '''
    # Extract the base file name without extension and add suffix
    base_name = os.path.splitext(os.path.basename(path))[0]
    file_name = f"{base_name}_scenario.pkl"

    # Define the folder path for scenarios
    file_folder = os.path.dirname(path)
    scenario_folder = os.path.join(file_folder, "scenarios")

    # Construct the full path to the scenario file
    path_to_file = os.path.join(scenario_folder, file_name)

    # Load the scenario object from the constructed file path
    scenario = util.load_interim_object(path_to_file)

    # Get indices for all steady-state periods
    ss_period_indices_all = get_ss_periods(scenario['ssd_periods_timestamps'], ss_detected.index)

    # Create a binary Series representing steady-state periods
    steady_state_series = create_binary_series(ss_period_indices_all)

    # Align the index of the result with the original timeframe
    steady_state_series.index = ss_detected.index

    return steady_state_series


def compute_evaluation_measures(actual_values, detected_values, measure_type):

     # Calculate the confusion matrix
    cm = confusion_matrix(actual_values, detected_values)
    # Extract the components of the confusion matrix, handle cases where confusion matrix is not 2x2
    if cm.size == 1:
        # All values are either 0 or 1
        if actual_values[0] == 0:
            TN, FP, FN, TP = cm[0], 0, 0, 0  # All true negatives
        else:
            TN, FP, FN, TP = 0, 0, 0, cm[0]  # All true positives
    else:
        # Confusion matrix is 2x2, unpack as usual
        TN, FP, FN, TP = cm.ravel()


    # Calculate evaluation metrics
    accuracy = accuracy_score(actual_values, detected_values)
    mcc = matthews_corrcoef(actual_values, detected_values)
    precision = precision_score(actual_values, detected_values, average=None, zero_division=0.0)
    recall = recall_score(actual_values, detected_values, average=None, zero_division=0.0)
    f1 = f1_score(actual_values, detected_values, average=None, zero_division=0.0)

    # Store the metrics in a dictionary

    metrics = {
     'measure_type': measure_type,
     'Confusion Matrix': cm.tolist(),  # Convert to list for easier display
     'TN': TN,
     'FP': FP,
     'FN': FN,
     'TP': TP,
     'Accuracy': accuracy,
     'Precision_0': precision[0] if len(precision) > 1 else precision[0],
     'Precision_1': precision[1] if len(precision) > 1 else 0,
     'Recall_0': recall[0] if len(recall) > 1 else recall[0],
     'Recall_1': recall[1] if len(recall) > 1 else 0,
     'F1_0': f1[0] if len(f1) > 1 else f1[0],
     'F1_1': f1[1] if len(f1) > 1 else 0,
     'MCC_phi': mcc
    }

    return metrics


def compare_actual_and_detected_steady_states(actual: pd.Series, detected: pd.Series, measure_type: str = "overall"):
    """
    Compares the actual steady-state periods with the detected steady-state periods by calculating precision, recall, F1 score,
    and optionally the Phi coefficient based on the provided measure type.

    Parameters:
    actual (pd.Series): A pandas Series representing the actual steady-state periods. Values should be either 0 or 1.
    detected (pd.Series): A pandas Series representing the detected steady-state periods. Values should be either 0 or 1.
    measure_type (str): Specifies which evaluation measures to calculate.
                        Acceptable values are 'overall', 'ss' (steady-state), or 'no_ss' (non-steady-state).
                        Default is 'overall'.

    Returns:
    dict: A dictionary containing the evaluation measures (precision, recall, F1, and Phi) based on the selected `measure_type`.

    Raises:
    ValueError: If the length of `actual` and `detected` Series do not match.
    TypeError: If `actual` or `detected` is not a pandas Series.
    ValueError: If `measure_type` is not one of 'overall', 'ss', or 'no_ss'.

    Notes:
    - Both `actual` and `detected` Series should be binary, containing only values 0 or 1.
    """

    # Validate input types
    if not isinstance(actual, pd.Series) or not isinstance(detected, pd.Series):
        raise TypeError("Both 'actual' and 'detected' must be pandas Series.")

    # Ensure both Series have the same length
    if len(actual) != len(detected):
        print(f"The length of 'actual' and 'detected' Series must be the same: {len(actual)} vs {len(detected)}.")
        if len(actual) > len(detected):
            actual = actual[actual.index.isin(detected.index)]
        else:
            detected = detected[detected.index.isin(actual.index)]

    # Validate measure_type
    if measure_type not in ['overall', 'ss', 'no_ss']:
        raise ValueError(f"Invalid measure_type: {measure_type}. Must be 'overall', 'ss', or 'no_ss'.")

    # Determine the subset of data to evaluate based on measure_type
    if measure_type == "overall":
        y_true = actual.tolist()
        y_pred = detected.tolist()

    elif measure_type == "ss":  # Steady state only
        y_true = actual[actual == 1].tolist()
        indices = actual[actual == 1].index
        y_pred = detected[indices].tolist()

    elif measure_type == "no_ss":  # Non-steady state only
        y_true = actual[actual == 0].tolist()
        indices = actual[actual == 0].index
        y_pred = detected[indices].tolist()

    # calculate evaluation measures
    results = compute_evaluation_measures(y_true, y_pred, measure_type)

    return results


def create_binary_series(ranges):
    """
    Creates a pandas Series of binary values (0 or 1) based on the provided index ranges.

    Parameters:
    ranges (list of lists): A list where each sublist contains two integers representing
                            the start and end indices (inclusive) where the value should be 1.

    Returns:
    pd.Series: A Series of 0s and 1s, where the indices within the specified ranges are 1.
    """
    # Determine the maximum index value for the Series
    max_value = max(end for _, end in ranges) + 1 if ranges else 0

    # Create a Series filled with 0
    binary_series = pd.Series(0, index=range(max_value))

    # Set the values within the specified ranges to 1
    for start, end in ranges:
        binary_series[start:end + 1] = 1  # +1 to include the end in the range

    return binary_series


def get_ss_periods(ss_period_timestamps, timeframe):
    """
    Converts steady-state period timestamps into their corresponding indices within a given timeframe.

    Parameters:
    ss_period_timestamps (list of lists): A list of lists where each sublist contains timestamps representing
                                          the start and end of steady-state periods.
    timeframe (pd.DatetimeIndex): The full index of timestamps for the entire timeframe.

    Returns:
    list of lists: A list of lists where each sublist contains indices corresponding to the steady-state periods.
    """
    ss_period_indices_all = []

    # Convert each period's timestamps to corresponding indices
    for ss_period in ss_period_timestamps:
        ss_period_indices = [timeframe.tz_localize(None).searchsorted(period, side='left') for period in ss_period]
        ss_period_indices_all.append(ss_period_indices)

    return ss_period_indices_all


def get_actual_steady_state_index(experiment) -> pd.Series:
    """
    Generates a Series indicating steady-state (1) and non-steady-state (0) periods based on simulated data.

    Parameters:
    date (pd.DataFrame): DataFrame that contains the timeframe index.

    Returns:
    pd.Series: A Series with indices corresponding to the timeframe, containing 1 for steady-state periods and 0 otherwise.
    """
    # Load the scenario with information about simulated steady-state periods
    file_name =  f"{os.path.basename(experiment['path']).split('.')[0]}_scenario.pkl"
    file_folder = os.path.dirname(experiment['path'])
    path_to_file = os.path.join(file_folder, file_name)
    scenario = util.load_interim_object(path_to_file)

    # Extract the timeframe index
    timeframe = experiment['ss_result'].index

    # Get indices for all steady-state periods
    ss_period_indices_all = get_ss_periods(scenario['ssd_periods_timestamps'], timeframe)

    # Create a binary Series representing steady-state periods
    steady_state_series = create_binary_series(ss_period_indices_all)

    # Align the index of the result with the original timeframe
    steady_state_series.index = timeframe

    return steady_state_series
