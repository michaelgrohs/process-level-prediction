import time
import sys

import os
from approach import utilities as util
from approach.evaluation import compare_actual_and_detected_steady_states, get_actual_steady_state_index
from main import multiprocessing
import pandas as pd
import approach.configurations as config


def evaluation_aggregated_new(all_experiments):

    evaluation_final_report_flattened = []

    for experiments_per_approach_setting in all_experiments:
        for experiment in experiments_per_approach_setting:
            actual_ss = get_actual_steady_state_index(experiment)
            detected_ss = experiment['ss_result']
            for evaluation_measure in ['overall']: #, 'ss', 'no_ss'
                evaluation_results = compare_actual_and_detected_steady_states(actual_ss, detected_ss, evaluation_measure)
                experiment = experiment | evaluation_results
                evaluation_final_report_flattened.append(experiment)

    evaluation_final_report_df = pd.DataFrame(evaluation_final_report_flattened)
    evaluation_final_report_df = evaluation_final_report_df.drop(['ss_result', 'ss_traces', 'ss_traces_per_period'], axis=1)

    return evaluation_final_report_df


def run_new_evaluation_experiment_1(scenario_folder, num_cores: int = config.NUM_CORES):
    approach_parameters = util.load_ssd_parameters_new(scenario_folder, config.DEFAULT_PARAMETER_DIR)

    results = multiprocessing(input_folder=scenario_folder, num_cores=num_cores, approach_parameters=approach_parameters)
    return results

def print_execution_duration(start_time, scenario_name=None):
    end_time = time.time()
    execution_duration = end_time - start_time
    formatted_time = util.format_time(execution_duration)
    if scenario_name:
        print(f"{scenario_name} - execution took {formatted_time}")
    else:
        print(f"Overall execution took {formatted_time}")


if __name__ == "__main__":
    num_cores = 170
    path_to_input_folder = "evaluation/experiment_1/input/"
    path_to_output_folder = "output/experiment_1/results/"
    scenario_folders = [
        "scenario_1",
        "scenario_2",
        "scenario_3",
        "scenario_4",
        "scenario_5",
        "scenario_6",
        "scenario_7",
        "scenario_8",
        "scenario_9",
        "scenario_10",
    ]

    # Timing: start
    start_time = time.time()  # Record the start time

    for scenario_name in scenario_folders:
        input_folder = os.path.join(path_to_input_folder, scenario_name)
        print(f"Working on {input_folder}")
        # TODO: rename function to run_evaluation_experiment_1
        results = run_new_evaluation_experiment_1(input_folder, num_cores)

        results_evaluated = evaluation_aggregated_new(results)
        os.makedirs(path_to_output_folder, exist_ok=True)

        save_results_to = os.path.join(path_to_output_folder, f'results_evaluation_{scenario_name}.csv')
        results_evaluated.to_csv(save_results_to, index=False)
        print(f"Results saved to the file '{save_results_to}'")
        print_execution_duration(start_time, scenario_name)

    # Record the end time, calculate the duration, and print the statement
    print_execution_duration(start_time)

    # Close the program
    sys.exit()

