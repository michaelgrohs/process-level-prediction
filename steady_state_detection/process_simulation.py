from os import mkdir

import ciw
import matplotlib.pyplot as plt
import random
import networkx as nx
import numpy as np
import pandas as pd
from multiprocessing import Pool, freeze_support, RLock, cpu_count
from tqdm import tqdm
from time import time
import os
from datetime import datetime
import approach.configurations as config
import pm4py
import pickle
from typing import List
from itertools import permutations, product
from tqdm import tqdm  # Import tqdm for the progress bar

from approach.utilities import format_time


def visualize_network(routing_matrix):
    graph = nx.DiGraph()

    for i, row in enumerate(routing_matrix):
        for j, prob in enumerate(row):
            if prob > 0:
                graph.add_edge(i, j, weight=prob)

    pos = nx.spring_layout(graph)
    labels = {i: f'Node {i}' for i in range(len(routing_matrix))}
    nx.draw(graph, pos, with_labels=True, labels=labels, font_weight='bold', node_size=700, node_color='skyblue',
            font_color='black', font_size=8, edge_color='black',
            width=[d['weight'] * 5 for (u, v, d) in graph.edges(data=True)])
    plt.show()

    return None

def generate_linear_routing_matrix(num_nodes: int = None):

    matrix = [[0] * num_nodes for _ in range(num_nodes)]  # Create an n x n matrix filled with zeros
    for i in range(num_nodes-1):
        matrix[i][i+1] = 1
    return matrix


def generate_random_routing_matrix(num_nodes: int = None) -> np.ndarray:
    """
    Generates a routing matrix representing the probabilities of routing from one node to another.

    Args:
        num_nodes (int): The number of nodes in the network.

    Returns:
        np.ndarray: A routing matrix where each row represents the probabilities of routing from one node to other nodes.

    Raises:
        ValueError: If num_nodes is not a positive integer.

    """
    if not num_nodes:
        num_nodes = random.randint(5, 20)

    # Check if num_nodes is a positive integer
    if not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError("Number of nodes must be a positive integer.")

    # Initialize a routing matrix with zeros
    routing_matrix = np.zeros((num_nodes, num_nodes))

    # Iterate over nodes to populate routing matrix
    for i in range(num_nodes - 1):
        # Determine available destination nodes
        available_indices = [x for x in range(num_nodes) if x > i]
        # Limit the number of possible destinations to 3 or the remaining nodes
        max_number_of_destination_nodes = min(3, len(available_indices))

        # Decide the number of non-zero values for the current node
        if max_number_of_destination_nodes > 1:
            num_nonzero = np.random.randint(1, max_number_of_destination_nodes)
            #num_nonzero = int(np.random.triangular(1, (max_number_of_destination_nodes+1)/2, max_number_of_destination_nodes))
        else:
            num_nonzero = 1

        # Choose random destination nodes
        available_indices = available_indices[:num_nonzero]
        indices = np.random.choice(available_indices, size=num_nonzero, replace=False)

        # Generate random values for the chosen destination nodes
        values = np.random.rand(num_nonzero)
        # Normalize values so they sum up to 1
        values /= np.sum(values)
        # Assign the normalized values to the routing matrix
        routing_matrix[i, indices] = np.round(values, 2)

    return routing_matrix.tolist()



def define_routing_matrix(option: str = 'zahoransky'):


    if option == 'simple_model':
        routing_matrix = [
            [0.0, 0.8, 0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    elif option == 'middle_model':
        routing_matrix = [
             [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    elif option == 'zahoransky':

        routing_matrix = [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    elif option.split('_')[0] == 'linear':
        num_nodes = int(option.split('_')[1])
        routing_matrix = generate_linear_routing_matrix(num_nodes)

    elif option.split('_')[0] == 'random':
        num_nodes = int(option.split('_')[1])
        routing_matrix = generate_random_routing_matrix(num_nodes)
    else:
        routing_matrix = generate_random_routing_matrix()

    #visualize_network(routing_matrix)

    return routing_matrix


def simulate_network(network, sim_time, seed: int = 1):
    recs_all = []
    ciw.seed(random.randint(0, 999999))
    Q = ciw.Simulation(network, tracker=ciw.trackers.SystemPopulation())
    Q.simulate_until_max_time(sim_time)
    recs = Q.get_all_records()
    recs_all.extend(recs)
    system_state = pd.DataFrame(Q.statetracker.history, columns=['time', 'state'])
    return recs_all, system_state



def extract_event_logs(recs_all):
    all_event_logs_df = []
    for recs in recs_all:
        event_log_df = create_event_log(recs)
        all_event_logs_df.append(event_log_df)

    return all_event_logs_df


def create_event_log(event_records):
    event_log = []

    for record in event_records:
        log_entry = {
            "CustomerID": record.id_number,
            "CustomerClass": record.customer_class,
            "ArrivalTime": record.arrival_date,
            "WaitingTime": record.waiting_time,
            "ServiceStartTime": record.service_start_date,
            "ServiceTime": record.service_time,
            "ServiceEndTime": record.service_end_date,
            "TimeBlocked": record.time_blocked,
            "ExitTime": record.exit_date,
            "Destination": record.destination,
            "QueueSizeAtArrival": record.queue_size_at_arrival,
            "QueueSizeAtDeparture": record.queue_size_at_departure,
            "ServerID": record.server_id,
            "RecordType": record.record_type
        }

        event_log.append(log_entry)

    event_log_df = pd.DataFrame(event_log)

    # Sort dataframe
    event_log_df['CustomerID'] = event_log_df['CustomerID'].astype(int)
    event_log_df['ExitTime'] = event_log_df['ExitTime'].astype(float)
    event_log_df = event_log_df.sort_values(by=["CustomerID", "ExitTime"])

    # Add activity name (node) for each event
    #event_log_df['Node'] = event_log_df.groupby('CustomerID').apply(calculate_node).reset_index(level=0, drop=True)
    event_log_df['Node'] = event_log_df.groupby('CustomerID')[['Destination']].apply(calculate_node).reset_index(level=0, drop=True)
    #event_log_df['Node'] = event_log_df.groupby('CustomerID', group_keys=False).apply(calculate_node).reset_index(level=0, drop=True)
    #event_log_df['Node'] = event_log_df.groupby('CustomerID', group_keys=False, include_groups=False).apply(calculate_node).reset_index(level=0, drop=True)
    #event_log_df['Node'] = event_log_df.groupby('CustomerID', group_keys=False).apply(lambda x: calculate_node(x)).reset_index(level=0, drop=True)

    return event_log_df


def calculate_node(group):
    control_flow = [1] + group['Destination'].tolist()[:-1]
    return pd.Series(control_flow, index=group.index)



def extract_case_features(event_logs):
    event_logs_grouped = event_logs.groupby('CustomerID')
    # extract key data per customer id
    data_record = []
    for customer_id, group in event_logs_grouped:
        case_arrival_time = group.ArrivalTime.min()
        case_completion_time = group.ExitTime.max()
        lead_time = case_completion_time - case_arrival_time
        control_flow = [1] + group.Destination.tolist()[:-1] #remove last destination "-1" and add the first one (always "0")
        server_id_per_node = group.ServerID.tolist()
        data_record.append([customer_id, case_arrival_time, case_completion_time, lead_time, control_flow, server_id_per_node])

    date_records_df = pd.DataFrame(data_record, columns=['case_id', 'arrival', 'departure', 'lead_time', 'control_flow', 'server_id_per_node'])

    return date_records_df


def calc_timeframe(data, period):
    # Crete timeframe given max and min time from the log and a given period
    min_time = 0
    max_time = int(np.ceil(pd.concat([data['arrival'], data['departure']]).agg(['max'])))
    timeframe = range(min_time, max_time+period, period)
    return timeframe

def calc_arrivals_or_departures(data, column_name, timeframe):
    # Group the column values into intervals defined by 'timeframe'
    grouped = pd.cut(data[column_name], bins=timeframe)

    # Count the number of values in each interval
    grouped_counts = grouped.value_counts()

    # Sort the grouped counts by the bin intervals
    sorted_grouped_counts = grouped_counts.sort_index()

    # Reset the index and set it to start from 1
    sorted_grouped_counts.reset_index(drop=True, inplace=True)

    # Increment the index by 1 to start from 1 instead of 0
    sorted_grouped_counts.index += 1

    return sorted_grouped_counts


def calc_active_cases_at_window_end(case_arrivals, case_departures):
    cases_active = (case_arrivals - case_departures).cumsum()
    return cases_active


def calc_average_lead_time(data, column_name, timeframe):
    # Group the column values into intervals defined by 'timeframe'
    grouped = pd.cut(data[column_name], bins=timeframe)

    # Calculate the average value for each interval
    grouped_mean = data.groupby(grouped)['lead_time'].mean()

    # Reset the index and set it to start from 1
    grouped_mean.reset_index(drop=True, inplace=True)

    # Increment the index by 1 to start from 1 instead of 0
    grouped_mean.index += 1

    return grouped_mean


def calc_active_servers(data, timeframe):

    grouped = pd.cut(data['arrival'], bins=timeframe)
    data_grouped = data.groupby(grouped)
    # extract key data per customer id
    result = []
    for window, group in data_grouped:
        big_list = [(control_flow, server_id) for _, row in group.iterrows() for control_flow, server_id in
                    zip(row['control_flow'], row['server_id_per_node'])]
        active_servers = len(set(big_list))
        result.append([window, active_servers])

    index = range(1, len(data_grouped)+1)
    values = [value for _, value in result]

    active_servers = pd.Series(values, index=index, name='active_servers')

    # active_servers_df = pd.DataFrame(result, columns=['Interval', 'Value']).set_index('Interval')
    # active_servers_df.reset_index(drop=True, inplace=True)
    # active_servers_df.index += 1
    return active_servers


def add_current_activity_executed(event_log_df):
    event_logs_grouped = event_log_df.groupby('CustomerID')

    # extract key data per customer id
    data_record = []
    for customer_id, group in event_logs_grouped:
        group['calc_activity_executed'] = [1] + group['Destination'][:-1].values.tolist()
        data_record.append(group)

    # Concatenate the list of modified groups back into a single DataFrame
    event_log_df_ext = pd.concat(data_record)

    return event_log_df_ext


def calculate_average_utilization(event_log_df, timeframe):

    # Add column with the name of the currently executed activity for each event
    event_log_df_ext = add_current_activity_executed(event_log_df)

    # Define groupping based on the timeframe
    groupping = pd.cut(event_log_df_ext['ServiceEndTime'], bins=timeframe)
    # Sum up service time for each period and server/activity
    event_log_df_ext_grouped = event_log_df_ext.groupby([groupping, "calc_activity_executed"]).agg({'ServiceTime': 'sum'})
    # Define all values by the duration of the period to can utilization as a ration
    event_log_df_ext_grouped = event_log_df_ext_grouped / timeframe.step
    # Calculate average utilization for each period
    average_utilization = event_log_df_ext_grouped.groupby("ServiceEndTime").mean()*100

    average_utilization.index =  range(1, len(timeframe) )
    new_column_names = {'ServiceTime': 'active_servers'}
    event_log_df_ext_grouped = average_utilization.rename(columns=new_column_names)

    return average_utilization


def create_time_series(event_log_df, system_state, windowing: int = 10):

    # Extract relevant time info per case
    features_per_case = extract_case_features(event_log_df)

    # Crete timeframe given max and min time from the log and a given period
    timeframe = calc_timeframe(features_per_case, windowing)

    # Calculate time series for process features
    case_arrivals = calc_arrivals_or_departures(features_per_case, 'arrival', timeframe)
    case_lead_time = calc_average_lead_time(features_per_case, 'departure', timeframe)
    case_active = calc_active_cases(system_state, timeframe)
    #active_servers = calc_active_servers(features_per_case, timeframe)
    active_servers = calculate_average_utilization(event_log_df, timeframe)

    # Combine all time series to a dataframe
    time_series_df = pd.concat([case_arrivals, case_active, case_lead_time, active_servers], axis=1)
    time_series_df.columns = ['case_arrivals', 'case_active', 'case_lead_time', 'active_servers']

    return time_series_df


def get_shock_moments(resilience_scenario):
    # Retrieve the list of periods
    periods = resilience_scenario['ssd_periods']
    n = len(periods)

    # Handle cases where the list has fewer than 2 periods
    if n < 2:
        return None

    # Initialize shock moments list
    shock_moments = []

    # Add the last value of the first period
    shock_moments.append(periods[0][1])

    # Loop through the middle periods and add alternating values
    for i in range(1, n - 1):
        shock_moments.append(periods[i][0])  # Start of current period
        shock_moments.append(periods[i][1])  # End of current period

    # Add the first value of the last period
    shock_moments.append(periods[-1][0])

    return shock_moments



def plot_time_series(time_series, scenario):

    norm_arrivals = scenario['normal_arrival_rate']
    pick_arrival = scenario['pick_arrival_rate']
    windowing = scenario['windowing']
    shock_moments = get_shock_moments(scenario)

    fig, axes = plt.subplots(nrows=len(time_series.columns), ncols=1, figsize=(10, 8))
    plt.title(f"Arrival rate (min): {norm_arrivals}, pick arrival (max): {pick_arrival}, windowing {windowing}")

    # Plot each column in a separate subplot
    for i, column in enumerate(time_series.columns):
        time_series[column].plot(ax=axes[i])
        axes[i].set_ylabel(column)
        if shock_moments is not None:
            for shock_moment in shock_moments:
                axes[i].axvline(x=shock_moment / windowing, color= 'r', linestyle='--')

    plt.tight_layout()
    folder_name = f"{scenario['folder_name']}/plot_sim_features"
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(folder_name + f"/event_log_{scenario['scenario_id']}.png")
    plt.show()
    plt.close()


def calc_active_cases(system_state, timeframe):

    # Group the column values into intervals defined by 'timeframe'
    grouped = pd.cut(system_state['time'], bins=timeframe)

    # Find the maximum value in each interval
    grouped_max = system_state.groupby(grouped).max()

    # Reset the index and set it to start from 1
    grouped_max.reset_index(drop=True, inplace=True)
    grouped_max.index += 1

    return grouped_max['state']



def create_network(resilience_scenario):

    # Define model (using a routing matrix)
    routing_matrix = resilience_scenario['routing_matrix']
    #if resilience_scenario['plot']:
    #    visualize_network(routing_matrix)

    num_nodes = len(routing_matrix)

    # Disruption in the arrivals
    Pi = ciw.dists.PoissonIntervals(
        rates=resilience_scenario['arrivals']['rates'],
        endpoints=resilience_scenario['arrivals']['endpoints'],
        max_sample_date=resilience_scenario['arrivals']['max_sample_date'] #TODO remove max_sample_date
    )
    dist_arrival = [Pi] + [None] * (num_nodes - 1)

    # Define disruption in the available resources through service rate
    #dist_service_rate = [resilience_scenario['service_rate']] * num_nodes
    dist_service_rate = resilience_scenario['service_rate']

    # Define disruption in the available resources through number of servers per node
    server_schedule_all = [resilience_scenario['server_schedule'] for _ in range(num_nodes)]

    network = ciw.create_network(
        arrival_distributions=dist_arrival,
        service_distributions=dist_service_rate,
        routing=routing_matrix,
        number_of_servers=server_schedule_all
    )

    return network



def define_max_duration_for_triangular_dist(option):


    if option == 'simple':
        max_duration = 8 # 2 1->2
    elif option == 'middle':
        max_duration = 10 # 4 2->3
    elif option == 'complex':
        max_duration = 12 # 6 3->6
    else:
        raise ValueError("Invalid max duration for the triangular dist. Choose 'simple', 'middle', or 'complex'.")

    return max_duration

def define_service_rates(routing_matrix, complexity: str = 'zahoransky'):
    num_nodes = len(routing_matrix)

    if complexity.split('_')[0] == 'deterministic':
        duration = int(complexity.split('_')[1])
        service_rates = [ciw.dists.Deterministic(value=duration)] * num_nodes

    elif complexity.split('_')[0] == 'triangular':
        triangular_shaping = [0, 1, 2]
        max_duration = define_max_duration_for_triangular_dist(complexity)
        parameters = [item * max_duration for item in triangular_shaping]
        service_rates = [ciw.dists.Triangular(lower=parameters[0], mode=parameters[1], upper=parameters[2])] * num_nodes

    elif complexity.split('_')[0] == 'normal':
        mu = float(complexity.split('_')[1])
        sigma = float(complexity.split('_')[2])
        service_rates = [ciw.dists.Normal(mean=mu, sd=sigma)] * num_nodes

    elif complexity.split('_')[0] == 'lognormal':
        mu = float(complexity.split('_')[1])
        sigma = float(complexity.split('_')[2])
        service_rates = [ciw.dists.Normal(mean=mu, sd=sigma)] * num_nodes

    elif complexity == 'zahoransky':
        service_rates = [
            ciw.dists.Lognormal(mean=0.2, sd=0.4),
            ciw.dists.Normal(mean=0.5, sd=0.1),
            ciw.dists.Normal(mean=0.6, sd=0.15),
            ciw.dists.Gamma(shape=0.9, scale=0.7),
            ciw.dists.Gamma(shape=0.8, scale=0.8),
            ciw.dists.Lognormal(mean=0.4, sd=0.4),
            ciw.dists.Lognormal(mean=0.4, sd=0.4),
            ciw.dists.Lognormal(mean=0.1, sd=0.5),
            ciw.dists.Gamma(shape=1.0, scale=0.5),
            ciw.dists.Lognormal(mean=0.25, sd=0.5),
            ciw.dists.Lognormal(mean=0.07, sd=0.3),
            ciw.dists.Normal(mean=0.8, sd=0.3),
            ciw.dists.Lognormal(mean=1.2, sd=0.4),
            ciw.dists.Normal(mean=1.0, sd=0.4),
            ciw.dists.Normal(mean=0.8, sd=0.3)]
    else:
        raise ValueError("Invalid service rates. Choose 'simple', 'middle', 'complex, or 'zahoransky'.")

    return service_rates


from typing import List


def transform_ss_periods_to_non_ss_periods(pairs: List[List[int]]) -> List[List[int]]:
    """
    Transforms a list of paired intervals by creating a new list where each element is a pair
    consisting of the end of the current interval and the start of the next interval.

    Parameters:
    - pairs (List[List[int]]): List of intervals, where each interval is a list of two integers [start, end].

    Returns:
    - List[List[int]]: Transformed list of intervals where each interval is [end of current, start of next].
    """

    # Create the transformed list by pairing the end of each interval with the start of the next interval
    transformed = [[pairs[i][1], pairs[i + 1][0]] for i in range(len(pairs) - 1)]

    return transformed


def create_change_rates(arrival_origin, arrival_target, step_size=0.05):
    # Determine the correct step direction
    if arrival_origin > arrival_target:
        step_size = -abs(step_size)
    else:
        step_size = abs(step_size)

    # Create the list using a loop and include the arrival_target as the last element
    result = []
    current_value = arrival_origin
    while (current_value <= arrival_target and step_size > 0) or (current_value >= arrival_target and step_size < 0):
        result.append(round(current_value, 2))
        current_value += step_size

    return result


def generate_endpoints(start, end, n):
    values = np.linspace(start, end, n)
    rounded_values = [int(value) for value in values]
    return rounded_values

def create_arrival_setting(arrival_origin, arrival_target, ss_periods, step_size = 0.05):

    change_rates = create_change_rates(arrival_origin, arrival_target, step_size)
    change_periods = transform_ss_periods_to_non_ss_periods(ss_periods)

    all_change_rates = []
    all_change_end_periods = []

    for i, change_period in enumerate(change_periods):
        change_end_periods = generate_endpoints(change_period[0], change_period[-1], len(change_rates)-1)
        all_change_end_periods.extend(change_end_periods)
        if i == 0:
            all_change_rates.extend(change_rates)
        else:
            all_change_rates.extend(change_rates[1:])
        change_rates.reverse()

    all_change_end_periods = all_change_end_periods + [ss_periods[-1][-1]]
    return all_change_rates, all_change_end_periods


def transform_ss_periods_to_timestamps(time_unit, start_time, ssd_periods):
    ssd_timestamps = [[(value * time_unit) + start_time for value in sublist] for sublist in ssd_periods]
    return ssd_timestamps

def define_process_disruption_scenario(
        change_type: str = "arrival_rate",
        number_of_changes: int = 1,
        simulation_periods: int = 500,
        target_arrival_rate: float = 0.6,
        origin_arrival_rate: float = 0.1,
        windowing: int = 72,
        process_model: str = 'zahoransky',
        process_complexity: str = 'zahoransky'):

    # define output dictionary
    disruption_scenario = {}

    disruption_scenario['change_type'] = change_type
    disruption_scenario['number_of_changes'] = number_of_changes
    disruption_scenario['process_model'] = process_model
    disruption_scenario['process_complexity'] = process_complexity
    disruption_scenario['windowing'] = windowing
    disruption_scenario['simulation_periods'] = simulation_periods
    disruption_scenario['pick_arrival_rate'] = target_arrival_rate
    disruption_scenario['normal_arrival_rate'] = origin_arrival_rate
    disruption_scenario['plot'] = True
    disruption_scenario['routing_matrix'] = define_routing_matrix(disruption_scenario['process_model'])
    disruption_scenario['service_rate'] = define_service_rates(disruption_scenario['routing_matrix'], process_complexity)
    disruption_scenario['server_schedule'] = 1
    disruption_scenario['time_unit'] = pd.Timedelta('1h')
    disruption_scenario['start_time'] = pd.Timestamp('2020-01-01')
    disruption_scenario['ssd_periods'] = calculate_steady_state_periods(number_of_changes, simulation_periods, windowing)
    disruption_scenario['ssd_periods_timestamps'] = transform_ss_periods_to_timestamps(disruption_scenario['time_unit'], disruption_scenario['start_time'], disruption_scenario['ssd_periods'])

    arrival_rates, endpoints = create_arrival_setting(origin_arrival_rate, target_arrival_rate, disruption_scenario['ssd_periods'])
    disruption_scenario['arrivals'] = {'rates': arrival_rates,  # or 0.4
                                       'endpoints': endpoints,
                                       'max_sample_date': endpoints[-1]}

    #shock_index: int = int(endpoints[0] // windowing + 1)
    #shock_index: int = 0
    disruption_scenario['disruption_moment'] = None


    return disruption_scenario



def set_resilience_scenario(disruption_type: str = 'no_disruption',
                            process_model: str = 'zahoransky',
                            process_complexity: str = 'zahoransky',
                            windowing: int = 40,
                            simulation_periods: int = 1000,
                            pick_arrival_rate: float = 0.6,
                            normal_arrival_rate: float = 0.3):

    # define general simulation parameters
    resilience_scenario = {'process_model': process_model,
                           'routing_matrix': None,
                           'process_complexity': process_complexity,
                           'service_rate': None,
                           'arrivals': {'rates': list,  # or 0.4
                                        'endpoints': list,
                                        'max_sample_date': int},
                           'server_schedule': 1,
                           'simulation_periods': simulation_periods,
                           'windowing': windowing,
                           'plot': True,
                           }

    resilience_scenario['routing_matrix'] = define_routing_matrix(resilience_scenario['process_model'])
    resilience_scenario['service_rate'] = define_service_rates(resilience_scenario['routing_matrix'], process_complexity)
    #visualize_network(resilience_scenario['routing_matrix'])

    # Define arrival rate parameters
    ciw_sim_end = windowing * simulation_periods
    if disruption_type == 'scenario_v1':
        resilience_scenario['disruption_type'] = 'scenario_v1'

        increase_start = ciw_sim_end / 10 * 4
        increase_duration_in_periods = simulation_periods / 5
        arrival_rates = list(np.arange(normal_arrival_rate, pick_arrival_rate, 0.05))
        increase_steps = int(increase_duration_in_periods / len(arrival_rates))
        endpoints = list(increase_start + windowing * int(increase_steps) * np.arange(len(arrival_rates)-1))+ [ciw_sim_end]

        ssd_periods = [ [0, endpoints[0]], [endpoints[-2], ciw_sim_end] ]



    elif disruption_type == 'scenario_v2':
        resilience_scenario['disruption_type'] = 'scenario_v2'
        increase_start: float = ciw_sim_end / 10 * 2
        increase_duration_in_periods = simulation_periods / 5
        increase_arrival_rates = list(np.arange(normal_arrival_rate, pick_arrival_rate, 0.05))
        increase_steps = int(increase_duration_in_periods / len(increase_arrival_rates)) + 1
        increase_end_points = increase_start + windowing * int(increase_steps) * np.arange(len(increase_arrival_rates))
        
        
        decrease_start: float = ciw_sim_end / 10 * 6
        decrease_duration_in_periods = simulation_periods / 5
        decrease_arrival_rates = list(np.arange(pick_arrival_rate, normal_arrival_rate, -0.05))
        decrease_steps = int(decrease_duration_in_periods / len(decrease_arrival_rates)) + 1
        decrease_end_points = decrease_start + windowing * int(decrease_steps) * np.arange(len(decrease_arrival_rates))

        endpoints = list(increase_end_points) + list(decrease_end_points) + [ciw_sim_end]
        arrival_rates = increase_arrival_rates + decrease_arrival_rates + [normal_arrival_rate]
        ssd_periods = [ [0, increase_start], [increase_end_points[-1], decrease_end_points[0]], [decrease_end_points[-1], ciw_sim_end] ]

    else:
        ValueError("Invalid disruption scenario. Choose 'disruption_in_arrivals_v1' or 'disruption_in_arrivals_v2'")

    resilience_scenario['arrivals'] = {'rates': arrival_rates,  # or 0.4
                                        'endpoints': endpoints,
                                        'max_sample_date': ciw_sim_end}
    resilience_scenario['ssd_periods'] = ssd_periods
    resilience_scenario['time_unit'] = pd.Timedelta('1h')
    resilience_scenario['start_time'] = pd.Timestamp('2020-01-01')

    ssd_timestamps = [[(value * resilience_scenario['time_unit']) + resilience_scenario['start_time'] for value in sublist] for sublist in ssd_periods]
    resilience_scenario['ssd_periods_timestamps'] = ssd_timestamps

    shock_index: int = int(endpoints[0] // windowing + 1)
    resilience_scenario['disruption_moment'] = shock_index
    return resilience_scenario


def save_scenario_info_pkl(scenario):

    folder_name = f"{scenario['folder_name']}/scenarios"
    os.makedirs(folder_name, exist_ok=True)

    with open(folder_name + f"/event_log_{scenario['scenario_id']}_scenario.pkl", 'wb') as file:
        # Save the dictionary to the file
        pickle.dump(scenario, file)


def time_series_clean_up(time_series, option: str = 'drop'):
    has_missing_values = time_series.isna().any(axis=1)

    if has_missing_values.any():
        na_counts = time_series.isna().sum().sum()

        if option == 'interpolation_linear':
            time_series = time_series.interpolate(method="linear")
        else:
            time_series = time_series.dropna()
        #print(f"Time series has {na_counts} missing values. Solution: {option} missing values")

    return time_series


def experiment(scenario):

    # DEFINE PROCESS NETWORK
    network = create_network(scenario)
    # SIMULATE PROCESS
    recs_all, system_state = simulate_network(network, scenario['arrivals']['max_sample_date'])
    # EXTRACT EVENT LOG FROM SIMULATED BEHAVIOR
    event_log_df = create_event_log(recs_all)

    # PRINT TIME SERIES
    if scenario['plot']:
        # CREATE TIME SERIES THAT RECORD SYSTEM-LEVEL PROCESS BEHAVIOR
        time_series = create_time_series(event_log_df, system_state, scenario['windowing'])
        time_series = time_series_clean_up(time_series, option='interpolation_linear')
        plot_time_series(time_series, scenario)

    # Rename columns to fit process mining standards (pm4py package)
    event_log = pm4py.format_dataframe(event_log_df, case_id='CustomerID', activity_key='Node', timestamp_key='ExitTime')

    # Change column format to data
    event_log['time:timestamp'] = scenario['start_time'] + event_log['time:timestamp'] * scenario['time_unit']

    return event_log


def main(scenario):
    # RUN SEVERAL EXPERIMENTS
    event_log_df = experiment(scenario)

    # Generate a unique file name using the scenario ID
    filename = f"{scenario['folder_name']}/event_log_{scenario['scenario_id']}.xes"

    # Write the event log with a unique name
    pm4py.write_xes(event_log_df, filename)

    return None


def create_output_folder(out_folder_path = config.DEFAULT_OUTPUT_DIR):
    # Create the folder
    folder_name = f"{out_folder_path}/experiment_1/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"Experiment folder '{folder_name}' created successfully.")

    return folder_name





def multi_processing(resilience_scenario, num_cores=None, n_runs=100):

    # Multiprocessing setup
    if num_cores is None:
        num_cores = cpu_count() - 2

    all_results = pd.DataFrame()
    arguments = [resilience_scenario] * n_runs

    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention

    with Pool(num_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        for results in tqdm(p.imap(experiment, arguments), desc="Running Experiments", total=len(arguments)):
            if results is not None:
                results['experiment_id'] = len(all_results) / len(results) + 1
                results['disruption_type'] = resilience_scenario['disruption_type']
                results['picked_arrival_rate'] = resilience_scenario['arrivals']['rates'][1]
                results['normal_arrival_rate'] = resilience_scenario['arrivals']['rates'][0]
                results['windowing'] = resilience_scenario['windowing']
                results['process_model'] = resilience_scenario['process_model']
                results['process_complexity'] = resilience_scenario['process_complexity']
                all_results = pd.concat([all_results, results], ignore_index=True)

    return all_results


def calculate_steady_state_periods(num_changes: int, simulation_period: int, scaling_factor: int) -> List[List[int]]:
    """
    Calculates alternating steady-state and non-steady-state periods within a simulation period and scales them by a factor.

    Parameters:
    - num_changes (int): Number of transition points between steady and non-steady states.
    - simulation_period (int): Total period of the simulation.
    - scaling_factor (int): Factor to scale each period.

    Returns:
    - List[List[int]]: A list of paired periods, each sublist containing scaled start and end points for each steady or non-steady state period.
    """

    # Calculate the length of each steady-state and non-steady-state period
    steady_state_period_length = int(simulation_period * 0.5 / num_changes)
    non_steady_state_period_length = int(simulation_period * 0.5 / (num_changes + 1))

    # Initialize the sequence with the starting point (0)
    period_sequence = [0]
    add_steady_state = True  # Flag to alternate between steady and non-steady periods

    # Generate sequence of period endpoints by alternating between steady and non-steady state lengths
    while period_sequence[-1] < simulation_period:
        if add_steady_state:
            next_value = period_sequence[-1] + non_steady_state_period_length
        else:
            next_value = period_sequence[-1] + steady_state_period_length
        if next_value > simulation_period:
            break
        period_sequence.append(next_value)
        add_steady_state = not add_steady_state  # Alternate between steady and non-steady state

    # Group consecutive elements in pairs to represent [start, end] of each period
    paired_periods = [period_sequence[i:i + 2] for i in range(0, len(period_sequence), 2)]

    # Scale each value in the paired periods by the scaling factor
    scaled_periods = [[time_point * scaling_factor for time_point in period] for period in paired_periods]

    return scaled_periods


def create_all_change_rates_combinations(values, threshold):
    # Generate all possible ordered pairs using permutations
    pairs = list(permutations(values, 2))
    # Filter pairs where the absolute difference between values is greater than the threshold
    return  [(a, b) for a, b in pairs if np.round(np.abs(a - b),1) > threshold]



def create_scenario_parameters_df(parameter_combinations):

    flattened_data = []
    for entry in parameter_combinations:
        index, path, param_values  = entry
        flattened_data.append([index, path]+list(param_values) )

    # Define the column names
    columns = ['id', 'path', "change_type", "number_of_changes", "simulation_periods", "target_arrival_rate",
        "origin_arrival_rate", "windowing", "process_model", "process_complexity"]

    # Create the dataframe
    df = pd.DataFrame(flattened_data, columns=columns)
    return df


def generate_scenario_parameters(
        arrival_rate_values,
        change_threshold,
        min_changes=1,
        max_changes=10,
        folder_name=os.getcwd()):

    # Generate (origin, target) arrival rate pairs
    rate_pairs = create_all_change_rates_combinations(arrival_rate_values, change_threshold)

    # Generate a list of possible values for number_of_changes
    change_counts = list(range(min_changes, max_changes + 1))

    # Generate all combinations of number_of_changes with each rate pair
    parameter_combinations = [
        (
            "arrival_rate",  # change_type
            changes,  # number_of_changes
            500,  # simulation_periods
            target,  # target_arrival_rate
            origin,  # origin_arrival_rate
            72,  # windowing
            'zahoransky',  # process_model
            'zahoransky'  # process_complexity
        )
        for changes, (origin, target) in product(change_counts, rate_pairs)
    ]

    scenario_settings_with_ids = [(idx, folder_name, setting) for idx, setting in enumerate(parameter_combinations, start=1)]
    scenario_settings_df = create_scenario_parameters_df(scenario_settings_with_ids)
    scenario_settings_df.to_csv(folder_name+"/scenario_parameters.csv")

    return scenario_settings_with_ids

def create_event_log_for_scenario(args):
    scenario_id, folder_name, scenario_setting = args
    # Define scenario based on settings
    scenario = define_process_disruption_scenario(*scenario_setting)
    scenario['folder_name'] = folder_name
    scenario['scenario_id'] = scenario_id
    try:
        main(scenario)
        save_scenario_info_pkl(scenario)
    except Exception as e:
        # Log the failed scenario ID and its configuration
        print(f"Failed scenario {scenario_id}, {scenario_setting}, error: {e}")


if __name__ == "__main__":
    start_time = time()

    # Scenario parameters
    number_of_cores = 20
    arrival_rate_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    change_threshold = 0.3
    changes_min = 1
    changes_max = 5

    # create output folder

    # Create parameters

    # cases_to_check = [157]
    # scenario_settings_with_ids = [tuple for tuple in scenario_settings if tuple[0] in cases_to_check]
    #
    # scenario_id, folder_name, scenario_setting = scenario_settings_with_ids[0]
    # scenario = define_process_disruption_scenario(*scenario_setting)
    # scenario['folder_name'] = folder_name
    # scenario['scenario_id'] = scenario_id
    # main(scenario)

    # Run scenarios in parallel with a controlled number of cores
    for i in range(10):
        folder_name = create_output_folder()
        scenario_settings = generate_scenario_parameters(arrival_rate_values, change_threshold, changes_min,
                                                         changes_max, folder_name)
        with Pool(processes=number_of_cores) as pool:
           # Wrap pool.map with tqdm to show progress
           for _ in tqdm(pool.imap(create_event_log_for_scenario, scenario_settings), total=len(scenario_settings), desc="Processing scenarios"):
               pass

    end_time = time()
    elapsed_time = end_time - start_time
    formatted_time = format_time(elapsed_time)
    print(f"Execution took {formatted_time}")
