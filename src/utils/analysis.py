import numpy as np


def analyze_history_length(
    neighbor_times_list: list, node_interact_times: np.ndarray, num_neighbors: int
):
    """Analyze the average, median, and maximum time differences between the current interaction
    times in node_interact_times and the historical interacions in neighbor_times_list.

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
    # Compute the time differences between the current interaction times and the historical interaction times
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
