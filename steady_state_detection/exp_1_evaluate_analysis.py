import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from typing import List
import re
import numpy as np
from typing import Literal, Optional
from tqdm import tqdm


def load_csv_as_dataframe(csv_file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        csv_file_path (str): The path to the CSV file.

    Returns:
        DataFrame: A pandas DataFrame containing the CSV data.
    """
    # Import the CSV file as a DataFrame
    df = pd.read_csv(csv_file_path)

    # Return the DataFrame
    return df


def define_evaluation_measure_to_be_selected(measure_focus, measure_type):
    measure_focus_ext = []
    for measure in measure_focus:
        if measure in ['Precision', 'Recall', 'F1']:
            if measure_type in ['overall', 'ss']:
                measure_focus_ext.append(measure+'_1')
            else:
                measure_focus_ext.append(measure+'_0')
        else:
            measure_focus_ext.append(measure)
    return measure_focus_ext

def create_subplots_for_process_feature(df, plot_name: str, legend_focus_on: str, output_folder, aggregation: str, measure_type: str, measure_focus: List[str]):
    """
    Create subplots for each unique value in the 'process feature' column.
    Each subplot will show 'precision' vs. 'recall' and color code by 'approach'.

    Args:
        df (DataFrame): The pandas DataFrame containing the data.
    """

    # Select evaluation measure to be rounded
    columns = ['Accuracy', 'Precision_0', 'Precision_1', 'Recall_0', 'Recall_1', 'F1_0', 'F1_1', 'MCC_phi']

    # Convert specified columns to numeric values, coercing errors to NaN
    df.loc[:, columns] = df.loc[:, columns].apply(pd.to_numeric, errors='coerce')

    # Round the numeric values in the specified columns to 4 decimal places
    df.loc[:, columns] = df.loc[:, columns].round(4)


    # Get unique process features
    if aggregation == 'no':
        feature_list = df['ssd_feature'].unique()
    else:
        feature_list = df['ssd_agg_option'].unique()

    # Determine the number of subplots needed
    num_subplots = len(feature_list)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(8, 4 * num_subplots))

    # If there's only one subplot, convert axes to a list for consistent iteration
    if num_subplots == 1:
        axes = [axes]

    # Loop through each process feature and create a plot
    for i, feature in enumerate(feature_list):
        # Filter data for the current process feature
        if aggregation == 'no':
            df_filtered_scope = df[df['ssd_feature'] == feature]
        else:
            df_filtered_scope = df[df['ssd_agg_option'] == feature]

        x_y_values = define_evaluation_measure_to_be_selected(measure_focus, measure_type)

        # Create a scatter plot with color-coded dots by 'approach'
        sns.scatterplot(ax=axes[i],
                        x=x_y_values[0],
                        y=x_y_values[1],
                        hue=legend_focus_on,
                        data=df_filtered_scope)

        # Set plot title and labels
        axes[i].set_title(f'Process Feature: {feature}, plot focus: {plot_name}, agg: {aggregation}, measure_type {measure_type}')
        axes[i].set_xlabel(measure_focus[0])
        axes[i].set_ylabel(measure_focus[1])

        # Set specific ticks for clarity
        axes[i].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Optionally, you can remove the grid for a cleaner look
        axes[i].grid(True)  # Set to False if you prefer no grid

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()
    # Save the plot in the output folder
    evaluation_folder = os.path.join(output_folder)
    os.makedirs(evaluation_folder, exist_ok=True)
    output_path = os.path.join(evaluation_folder, f'plot_{legend_focus_on}_{plot_name}_agg_{aggregation}_{measure_type}_{'_'.join(measure_focus)}.png')
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)



def plot_evaluation_results(df, output_folder):

    # measure_types = ['overall', 'ss', 'no_ss']
    measure_types = ['overall']

    aggregation_options = ['no', 'yes']
    # aggregation_options = ['no']

    measure_focuses = [['Precision', 'Recall'], ['Accuracy', 'MCC_phi']]

    for measure_focus in measure_focuses:
        save_results_to_folder = os.path.join(output_folder, 'evaluation_plots')
        os.makedirs(save_results_to_folder, exist_ok=True)

        run_per_window_size = False
        run_per_approach = False

        for measure_type in measure_types:
            df_scope_main = df[df['measure_type'] == measure_type]
            for aggregation in aggregation_options:
                if aggregation == 'no':
                    print(
                        f"Work in progress: output_folder {save_results_to_folder}, measure_type {measure_type}, aggregation {aggregation}")
                    df_scope = df_scope_main[df_scope_main['ssd_agg_option'] == 'no']

                    # Plot for each process feature: all results for each approach and each window size
                    create_subplots_for_process_feature(df_scope, 'all_results', "funcname", save_results_to_folder, aggregation,
                                                        measure_type, measure_focus)

                    # Plot for each process feature and approach: all results each window size
                    if run_per_window_size:
                        for window in df_scope.window_step.unique():
                            create_subplots_for_process_feature(df_scope[df_scope.window_step == window], window,
                                                                "funcname", save_results_to_folder, aggregation, measure_type,
                                                                measure_focus)

                    # Plot for each process feature and window size: all results for each approach
                    if run_per_approach:
                        for approach in df_scope.funcname.unique():
                            create_subplots_for_process_feature(df_scope[df_scope.funcname == approach], approach,
                                                                "window_step", save_results_to_folder, aggregation, measure_type,
                                                                measure_focus)
                else:
                    df_scope = df_scope_main[df_scope_main['ssd_agg_option'] != 'no']

                    # Plot for each process feature: all results for each approach and each window size
                    create_subplots_for_process_feature(df_scope, 'all_results', "funcname", save_results_to_folder, aggregation,
                                                        measure_type, measure_focus)

                    # Plot for each process feature and approach: all results each window size
                    if run_per_window_size:
                        for window in df_scope.window_step.unique():
                            create_subplots_for_process_feature(df_scope[df_scope.window_step == window], window,
                                                                "funcname",
                                                                save_results_to_folder, aggregation, measure_type, measure_focus)

                    # Plot for each process feature and window size: all results for each approach
                    if run_per_approach:
                        for approach in df_scope.funcname.unique():
                            create_subplots_for_process_feature(df_scope[df_scope.funcname == approach], approach,
                                                                "window_step", save_results_to_folder, aggregation, measure_type,
                                                                measure_focus)


def report_evaluation_per_feature(df):


    #ssd_features = ['active_cases', 'avg_lead_time']

    ssd_features = ['case_arrivals', 'case_completions', 'active_cases', 'avg_lead_time', 'executed_events', 'agg_feature']


    filtering = (df['ssd_agg_option'] == 'no') & (df['measure_type'] == 'overall') & (df['ssd_feature'].isin(ssd_features))
    filtered_df = df[filtering].copy()
    filtered_df['MCC_plus_Accuracy'] = filtered_df['MCC_phi'] + filtered_df['Accuracy']
    result_df = filtered_df.loc[filtered_df.groupby(['funcname', 'ssd_feature'])['MCC_plus_Accuracy'].idxmax()]
    result_df = result_df.drop(columns=['MCC_plus_Accuracy'])

    # Pivot the result_df to create the desired multilevel column table
    #pivot_df = result_df.pivot(index='funcname', columns='ssd_feature', values=['Accuracy', 'MCC_phi'])
    pivot_df = result_df.pivot(index='ssd_feature', columns='funcname', values=['Accuracy', 'MCC_phi'])

    # Optional: Sort the columns to have a clear structure
    pivot_df = pivot_df.sort_index(axis=1, level=[1, 0])

    # Round the values to two decimal places
    pivot_df = pivot_df.round(2)

    # Convert the DataFrame to a LaTeX table format
    latex_table = pivot_df.to_latex(multicolumn=True, multicolumn_format='c', float_format="%.2f")

    # Print the LaTeX table
    print(latex_table)

    return None


def report_evaluation_per_aggregation(df):

    filtering = (df['ssd_agg_option'] != 'no') & (df['measure_type'] == 'overall')
    filtered_df = df[filtering].copy()
    filtered_df['MCC_plus_Accuracy'] = filtered_df['MCC_phi'] + filtered_df['Accuracy']
    result_df = filtered_df.loc[filtered_df.groupby(['funcname', 'ssd_agg_option'])['MCC_plus_Accuracy'].idxmax()]
    result_df = result_df.drop(columns=['MCC_plus_Accuracy'])

    # Pivot the result_df to create the desired multilevel column table
    #pivot_df = result_df.pivot(index='funcname', columns='ssd_agg_option', values=['Accuracy', 'MCC_phi'])
    pivot_df = result_df.pivot(index='ssd_agg_option', columns='funcname', values=['Accuracy', 'MCC_phi'])

    # Optional: Sort the columns to have a clear structure
    pivot_df = pivot_df.sort_index(axis=1, level=[1, 0])

    # Round the values to two decimal places
    pivot_df = pivot_df.round(2)

    # Convert the DataFrame to a LaTeX table format
    latex_table = pivot_df.to_latex(multicolumn=True, multicolumn_format='c', float_format="%.2f")

    # Print the LaTeX table
    print(latex_table)

    return None


def add_log_id(df):
    ids = [re.search(r'log_(\d+)', log).group(1) for log in df['path'].values]
    df['id'] = np.array(ids, dtype=int)
    return df


def add_scenario_info(df, save_to_folder):

    log_scenarios = pd.read_csv(os.path.join(save_to_folder, 'scenario_parameters.csv'))
    rel_columns = ['id', 'number_of_changes', 'target_arrival_rate', 'origin_arrival_rate']
    df = pd.merge(df, log_scenarios[rel_columns], on='id', how='left')

    df['change_gap'] = np.round(np.abs(df['origin_arrival_rate'] - df['target_arrival_rate']),1)
    df['from_to'] = str(df['origin_arrival_rate']) + "->" + str(df['target_arrival_rate'])
    return df.copy()

def filter_data(df, ssd_features, ssd_agg_option=False):
    if ssd_agg_option:
        filtering = ((df['ssd_agg_option'] != 'no')
                     & (df['measure_type'] == 'overall')
                     & (df['ssd_feature'].isin(ssd_features)))
    else:
        filtering = ((df['ssd_agg_option'] == 'no')
                     & (df['measure_type'] == 'overall')
                     & (df['ssd_feature'].isin(ssd_features)))
    filtered_df = df[filtering].copy()
    return filtered_df

def report_evaluation_results(df, path_to_results_folder, ssd_agg_option=False, eval_measure='MCC_phi', save_interim=False):

    if ssd_agg_option:
        grouping_by = 'ssd_agg_option'
    else:
        grouping_by = 'ssd_feature'
    # function parameters
    ssd_features = ['case_arrivals', 'case_completions', 'active_cases', 'avg_lead_time', 'executed_events', 'agg_feature']
    non_par_columns = ['path', 'window_step', 'funcname', 'ssd_agg_option', 'ssd_feature',
                       'measure_type', 'Confusion Matrix', 'TN', 'FP', 'FN', 'TP', 'Accuracy',
                       'Precision_0', 'Precision_1', 'Recall_0', 'Recall_1', 'F1_0', 'F1_1',
                       'MCC_phi', 'MCC_plus_Accuracy', 'id', 'number_of_changes', 'target_arrival_rate', 'origin_arrival_rate',
                       'change_gap', 'from_to']

    # Preprocessing
    df = add_log_id(df)
    # todo: do we really need this step?
    df_ext = add_scenario_info(df, path_to_input_folder)

    # Analysis per process feature
    df_filtered = filter_data(df_ext, ssd_features, ssd_agg_option)

    # Main loop
    results_dict = []
    for id, group in df_filtered.groupby(['funcname', grouping_by]):
        group_rel = group.dropna(axis=1)

        approach_parameters = [col for col in group_rel.columns.tolist() if col not in non_par_columns]

        best_par = group_rel.groupby(approach_parameters).agg({eval_measure: 'mean'})[eval_measure].idxmax()
        best_val = group_rel.groupby(approach_parameters).agg({eval_measure: 'mean'})[eval_measure].max()
        record = {'method': id[0],
                  'feature': id[1],
                  f'{eval_measure}_best_val': best_val,
                  f'{eval_measure}_best_par': best_par}
        results_dict.append(record)

    results_df = pd.DataFrame(results_dict)
    evaluation_column = f'{eval_measure}_best_val'
    results_df.sort_values(by=[evaluation_column], ascending=False, inplace=True)
    if save_interim:
        path_to_file = os.path.join(path_to_results_folder, f'evaluation_report_best_agg_{ssd_agg_option}_{eval_measure}_parameters.csv')
        results_df.to_csv(path_to_file, index=True)

    pivot_df = results_df.pivot(index='feature', columns='method', values=f'{eval_measure}_best_val')

    # Optional: Sort the columns to have a clear structure
    pivot_df = pivot_df.sort_index(axis=1, level=[1, 0])

    # Round the values to two decimal places
    pivot_df = pivot_df.round(2)

    if save_interim:
        path_to_file = os.path.join(path_to_results_folder, f'evaluation_report_best_agg_{ssd_agg_option}_{eval_measure}.csv')
        pivot_df.to_csv(path_to_file, index=True)
        print(f"Results are save to the file {path_to_file}")

    return pivot_df

def build_pivot_table(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a pivot-style table with fixed feature order and value columns.
    Computes average and standard deviation and optionally saves the result.
    """

    feature_order = [
        "active_cases",
        "avg_lead_time",
        "case_completions",
        "kernel",
        "and",
        "median",
        "or",
    ]

    value_cols = [
        "apply_ssd_using_rolling_windows",
        "apply_ssd_cumsum",
        "apply_ssd_by_rhinehart_2013",
        "apply_ssd_ed_pelt_with_transitions",
    ]

    rename_map = {
        "apply_ssd_using_rolling_windows": "RW",
        "apply_ssd_cumsum": "CS",
        "apply_ssd_by_rhinehart_2013": "VF",
        "apply_ssd_ed_pelt_with_transitions": "EDP",
    }

    out = (
        df.groupby("feature")[value_cols]
          .agg(["mean", "std"])
    )

    # Flatten column names
    out.columns = [
        f"Average of {rename_map[col]}" if stat == "mean"
        else f"Std of {rename_map[col]}"
        for col, stat in out.columns
    ]

    # Enforce feature order
    out = out.reindex(feature_order)

    # Optional save
    if save_path is not None:
        out.to_csv(save_path, index=True)

    return out


if __name__ == "__main__":
    path_to_input_folder = "evaluation/experiment_1/input/"

    path_to_results_folder = "output/experiment_1/results/"
    #path_to_results_folder = "evaluation/experiment_1/results_replicated/"
    #path_to_results_folder = "evaluation/experiment_1/results_paper_testing/"

    # Specify the path to the CSV file
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

    measure = 'MCC_phi'
    # Generate the list with os.path.join
    results_all_phi = pd.DataFrame()

    for i, scenario_name in tqdm(enumerate(scenario_folders, start=1), desc=f"Load results data: "):
        path_to_results_file = os.path.join(path_to_results_folder, f"results_evaluation_{scenario_name}.csv")
        df = pd.read_csv(path_to_results_file, low_memory=False)

        r1 = report_evaluation_results(df, path_to_results_folder, ssd_agg_option=False, eval_measure=measure, save_interim=False)
        r2 = report_evaluation_results(df, path_to_results_folder, ssd_agg_option=True, eval_measure=measure, save_interim=False)
        result = pd.concat([r1, r2])
        result["experiment_id"] = i
        results_all_phi = pd.concat([results_all_phi, result])

    results_all_phi.to_csv(os.path.join(path_to_results_folder, f'evaluation_report_aggregated_mcc.csv'))
    build_pivot_table(results_all_phi, os.path.join(path_to_results_folder, f'evaluation_report_aggregated_mcc_analysis.csv'))

    sys.exit()
