import numpy as np

from src.utils.analysis import analyze_history_length


def test_analyze_history_length():
    """Test the analyze_history_length function from src/utils/analysis.py."""
    # analyze_history_length(neighbor_times_list: list, node_interact_times: list, num_neighbors: int)
    # Test case 1
    neighbor_times_list = [np.array([0, 1, 3, 5]), np.array([2, 6])]
    node_interact_times = np.array([10, 20])
    num_neighbors = 3
    (
        avg_time_diff,
        median_time_diff,
        max_time_diff,
        num_temporal_neighbors,
    ) = analyze_history_length(neighbor_times_list, node_interact_times, num_neighbors)
    # Time diff = [[9, 7, 5], [nan, 18, 14]]
    # avg_time_diff = [7, 16]
    # median_time_diff = [7, 16]
    # max_time_diff = [9, 18]
    # num_temporal_neighbors = [3, 2]
    assert np.array_equal(avg_time_diff, np.array([7, 16]))
    assert np.array_equal(median_time_diff, np.array([7, 16]))
    assert np.array_equal(max_time_diff, np.array([9, 18]))
    assert np.array_equal(num_temporal_neighbors, np.array([3, 2]))

    # Test case 2
    neighbor_times_list = [np.array([2, 4]), np.array([1, 3, 5])]
    node_interact_times = np.array([7, 19])
    num_neighbors = 4
    (
        avg_time_diff,
        median_time_diff,
        max_time_diff,
        num_temporal_neighbors,
    ) = analyze_history_length(neighbor_times_list, node_interact_times, num_neighbors)
    # Time diff = [[nan, nan, 5, 3], [nan, 18, 16, 14]]
    # avg_time_diff = [4, 16]
    # median_time_diff = [4, 16]
    # max_time_diff = [5, 18]
    # num_temporal_neighbors = [2, 3]
    assert np.array_equal(avg_time_diff, np.array([4, 16]))
    assert np.array_equal(median_time_diff, np.array([4, 16]))
    assert np.array_equal(max_time_diff, np.array([5, 18]))
    assert np.array_equal(num_temporal_neighbors, np.array([2, 3]))

    # Test case 3
    neighbor_times_list = [np.array([1, 3, 5, 9]), np.array([2, 4, 6, 11])]
    node_interact_times = np.array([10, 20])
    num_neighbors = 2
    (
        avg_time_diff,
        median_time_diff,
        max_time_diff,
        num_temporal_neighbors,
    ) = analyze_history_length(neighbor_times_list, node_interact_times, num_neighbors)
    # Time diff = [[5, 1], [14, 9]]
    # avg_time_diff = [3, 11.5]
    # median_time_diff = [3, 11.5]
    # max_time_diff = [5, 14]
    # num_temporal_neighbors = [2, 2]
    assert np.array_equal(avg_time_diff, np.array([3, 11.5]))
    assert np.array_equal(median_time_diff, np.array([3, 11.5]))
    assert np.array_equal(max_time_diff, np.array([5, 14]))
    assert np.array_equal(num_temporal_neighbors, np.array([2, 2]))
