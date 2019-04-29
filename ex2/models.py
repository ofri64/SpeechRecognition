import numpy as np


def predict_one_nn(train_data, test_sample, distance_metric):
    sorted_distances = sorted(train_data, key=lambda x: distance_metric(x[0], test_sample))
    _, predicted_label = sorted_distances[0]

    return predicted_label


def d(x1, x2, i, j):
    return np.linalg.norm(x1[:, i] - x2[:, j])


def DTW(x1, x2):
    x1_num_time_stamps = x1.shape[1] - 1
    x2_num_time_stamps = x1.shape[1] - 1
    return recursive_DTW(x1, x2, x1_num_time_stamps, x2_num_time_stamps, {})


def recursive_DTW(x1, x2, i, j, memo):
    # stopping criterion
    if i < 0 or j < 0:
        return np.inf
    if i == 0 and j == 0:
        return d(x1, x2, 0, 0)

    if (i, j) not in memo:
        memo[(i, j)] = d(x1, x2, i, j) + min(recursive_DTW(x1, x2, i-1, j-1, memo),
                                             recursive_DTW(x1, x2, i, j-1, memo),  recursive_DTW(x1, x2, i-1, j, memo))
    return memo[(i, j)]
