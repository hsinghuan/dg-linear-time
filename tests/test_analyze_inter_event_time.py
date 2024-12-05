import numpy as np

from src.utils.analysis import analyze_inter_event_time


def test_analyze_inter_event_time():
    """Test the analyze_inter_event_time function from src/utils/analysis.py."""
    # analyze_history_length(neighbor_times_list: list, node_interact_times: list, num_neighbors: int)
    # Test case 1
    neighbor_times_list = [np.array([0, 1, 3, 5]), np.array([2, 6])]
    node_interact_times = np.array([10, 20])
    avg_inter_event_time, median_inter_event_time, std_inter_event_time = analyze_inter_event_time(
        neighbor_times_list, node_interact_times
    )
    # inter-event times = [[1, 2, 2, 5], [4, 14]]
    # node_i_avg_inter_event_time = [2.5, 9]
    # avg_inter_event_time = 5.75
    # median_inter_event_time = 5.5
    # std_inter_event_time = 3.25
    assert np.array_equal(avg_inter_event_time, 5.75)
    assert np.array_equal(median_inter_event_time, 5.5)
    assert np.array_equal(std_inter_event_time, 3.25)

    # Test case 2
    neighbor_times_list = [np.array([2, 4]), np.array([1, 3, 5])]
    node_interact_times = np.array([7, 19])
    avg_inter_event_time, median_inter_event_time, std_inter_event_time = analyze_inter_event_time(
        neighbor_times_list, node_interact_times
    )
    # inter-event times = [[2, 3], [2, 2, 14]]
    # node_i_avg_inter_event_time = [2.5, 6]
    # avg_inter_event_time = 4.25
    # median_inter_event_time = 2.25
    # std_inter_event_time = 1.75
    assert np.array_equal(avg_inter_event_time, 4.25)
    assert np.array_equal(median_inter_event_time, 2.25)
    assert np.array_equal(std_inter_event_time, 1.75)

    # Test case 3
    neighbor_times_list = [np.array([1, 3, 5, 9]), np.array([2, 4, 6, 11])]
    node_interact_times = np.array([10, 20])
    avg_inter_event_time, median_inter_event_time, std_inter_event_time = analyze_inter_event_time(
        neighbor_times_list, node_interact_times
    )
    # inter-event times = [[2, 2, 4, 1], [2, 2, 5, 9]]
    # node_i_avg_inter_event_time = [2.25, 4.5]
    # avg_inter_event_time = 3.375
    # median_inter_event_time = 2.75
    # std_inter_event_time = 1.125
    assert np.array_equal(avg_inter_event_time, 3.375)
    assert np.array_equal(median_inter_event_time, 2.75)
    assert np.array_equal(std_inter_event_time, 1.125)
