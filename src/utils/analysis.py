import numpy as np


def analyze_target_historical_event_time_diff(
    neighbor_times_list: list, node_interact_times: np.ndarray, num_neighbors: int
):
    """Analyze the average, median, and maximum time differences between the target edge
    interaction times in node_interact_times and the historical edge interaction event times in
    neighbor_times_list.

    Also measure the number of temporal neighbors for each node. Note that we only consider the
    most recent num_neighbors neighbors for each node.
    :param neighbor_times_list: list of ndarrays of neighbor interaction times for each node
    :param node_interact_times: ndarray, node interaction times for each node in the current batch
    :param num_neighbors: int, number of temporal neighbors to consider for each node
    :return avg_time_diff: ndarray, shape (batch_size,), average time differences between the
        current interaction time and the historical interaction times
    :return median_time_diff: ndarray, shape (batch_size,), median time differences between the
        current interaction times and the historical interaction times
    :return max_time_diff: ndarray, shape (batch_size,), maximum time differences between the
        current interaction times and the historical interaction times
    :return num_temporal_neighbors: ndarray, shape (batch_size,), number of temporal neighbors for
        each node
    """
    # Compute the time differences between the target edge interaction times and the historical edge interaction times
    # Initialize a ndarray of shape (batch_size, num_neighbors) to np.nan
    time_diffs = np.full((len(node_interact_times), num_neighbors), np.nan)
    num_temporal_neighbors = np.full(len(node_interact_times), np.nan)
    for i, neighbor_times in enumerate(neighbor_times_list):
        # Only consider the most recent num_neighbors neighbors
        neighbor_times = neighbor_times[-num_neighbors:]
        num_temporal_neighbors[i] = len(neighbor_times)
        if len(neighbor_times) > 0:
            time_diffs[i, -len(neighbor_times) :] = node_interact_times[i] - neighbor_times
    # Compute the average, median, and maximum time differences
    avg_time_diffs = np.nanmean(time_diffs, axis=1)
    median_time_diffs = np.nanmedian(time_diffs, axis=1)
    max_time_diffs = np.nanmax(time_diffs, axis=1)

    return avg_time_diffs, median_time_diffs, max_time_diffs, num_temporal_neighbors


def analyze_inter_event_time(
    neighbor_times_list: list,
    node_interact_times: np.ndarray,
):
    """Compute the average inter-event time between two consecutive interactions for a target
    node's history and then average across nodes."""
    # avg_inter_event_time = 0
    # median_inter_event_time = 0
    # total_num = 0
    avg_inter_event_time_list = []
    median_inter_event_time_list = []
    for i, neighbor_times in enumerate(neighbor_times_list):
        neighbor_times = np.append(neighbor_times, node_interact_times[i])
        # calculate the inter-event time (difference between adjacent elements)
        inter_event_times = np.diff(neighbor_times)
        # assert inter event times are non-negative
        assert np.all(inter_event_times >= 0)
        node_i_avg_inter_event_time = np.mean(inter_event_times)
        node_i_median_inter_event_time = np.median(inter_event_times)
        if not np.isnan(
            node_i_avg_inter_event_time
        ):  # will be nan if there were no temporal neighbors (historical interactions)
            # avg_inter_event_time += node_i_avg_inter_event_time
            avg_inter_event_time_list.append(node_i_avg_inter_event_time)
            # median_inter_event_time += node_i_median_inter_event_time
            # total_num += 1
        if not np.isnan(node_i_median_inter_event_time):
            median_inter_event_time_list.append(node_i_median_inter_event_time)

    # avg_inter_event_time /= total_num
    avg_inter_event_time = np.mean(avg_inter_event_time_list)
    std_inter_event_time = np.std(avg_inter_event_time_list)
    # median_inter_event_time = np.median(median_inter_event_time_list)
    median_inter_event_time = np.mean(median_inter_event_time_list)
    # median_inter_event_time /= total_num
    return avg_inter_event_time, median_inter_event_time, std_inter_event_time
