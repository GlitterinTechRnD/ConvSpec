import copy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import *
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Any, Optional


def custom_train_test_split(x, y, test_size, method='KS'):
    if method == 'KS':
        distance = cdist(x, x)

    if method == 'SPXY':
        y = np.expand_dims(y, axis=-1)

        distance_x = cdist(x, x)
        distance_y = cdist(y, y)

        distance_x /= distance_x.max()
        distance_y /= distance_y.max()

        distance = distance_x + distance_y
    if method == 'ManualDivision':
        X_train = x[:int(len(x) * test_size)]
        y_train = y[:int(len(y) * test_size)]
        X_test = x[int(len(x) * test_size):]
        y_test = y[int(len(y) * test_size):]
        return X_train, X_test, y_train, y_test

    def max_min_distance_split(distance, train_size):
        i_train = []
        i_test = [i for i in range(distance.shape[0])]

        first_2_points = np.unravel_index(np.argmax(distance), distance.shape)

        i_train.append(first_2_points[0])
        i_train.append(first_2_points[1])

        i_test.remove(first_2_points[0])
        i_test.remove(first_2_points[1])

        for _ in range(train_size - 2):
            max_min_dist_idx = np.argmax(np.min(distance[i_train, :], axis=0))

            i_train.append(max_min_dist_idx)
            i_test.remove(max_min_dist_idx)

        return i_train, i_test

    if 0 < test_size < 1:
        test_size = int(x.shape[0] * test_size)
    index_train, index_test = max_min_distance_split(distance, x.shape[0] - test_size)
    x_train, x_test, y_train, y_test = x[index_train], x[index_test], y[index_train], y[index_test]

    return x_train, x_test, y_train.reshape(-1, ), y_test.reshape(-1, )


def move_avg(X_train, X_test, y_train, y_test, window_size=11):
    if window_size % 2 == 0:
        raise ValueError('The window_size parameter in move_avg preprocessing must be an odd number')

    def apply_move_avg(x):
        x_ma = copy.deepcopy(x)
        for i in range(x.shape[0]):
            out0 = np.convolve(x_ma[i], np.ones(window_size, dtype=int), 'valid') / window_size
            r = np.arange(1, window_size - 1, 2)
            start = np.cumsum(x_ma[i, :window_size - 1])[::2] / r
            stop = (np.cumsum(x_ma[i, :-window_size:-1])[::2] / r)[::-1]
            x_ma[i] = np.concatenate((start, out0, stop))
        return x_ma

    X_train_ma = apply_move_avg(X_train)
    X_test_ma = apply_move_avg(X_test)
    return X_train_ma, X_test_ma, y_train, y_test


def remove_high_variance_and_normalize(X_train, X_test, y_train, y_test, remove_feat_ratio):
    variances = np.var(X_train, axis=0)
    sorted_indices = np.argsort(variances)
    num_features_to_remove = int(X_train.shape[1] * remove_feat_ratio)
    features_to_remove = sorted_indices[-num_features_to_remove:]
    X_train = np.delete(X_train, features_to_remove, axis=1)
    X_test = np.delete(X_test, features_to_remove, axis=1)
    for i in range(X_train.shape[1]):
        min_val = np.min(X_train[:, i])
        max_val = np.max(X_train[:, i])
        X_train[:, i] = (X_train[:, i] - min_val) / (max_val - min_val)
        X_test[:, i] = (X_test[:, i] - min_val) / (max_val - min_val)
    return X_train, X_test, y_train, y_test


def pca(X_train, X_test, y_train, y_test, n_components=2, random_state=42):
    """
    Principal Component Analysis (PCA)

    :param x: shape (n_samples, n_features)
    :param n_components: The number of principal components to retain
    :param scale: scale the principal components by the standard deviation of the corresponding eigenvalue
    """

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, y_train, y_test


def get_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()

    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'r': pearsonr(y_true, y_pred)[0]
    }
    return metrics
