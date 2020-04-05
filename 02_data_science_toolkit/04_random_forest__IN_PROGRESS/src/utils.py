from typing import List, Tuple, Union

import numpy as np


def group_gini_index(samples: np.array) -> float:
    n_samples = samples.shape[0]
    _, n_groups = np.unique(samples, return_counts=True)
    return 1 - sum(map(lambda x: (x/n_samples)**2, n_groups))


def groups_gini_index(first: np.array, second: np.array) -> float:
    n_first = first.shape[0]
    n_second = second.shape[0]
    n_total = n_first + n_second
    return group_gini_index(first) * n_first / n_total + \
           group_gini_index(second) * n_second / n_total


def find_split(X: np.array, y: np.array):
    n_samples, n_features = X.shape
    split_feature, split_value, best_gini, X_left, X_right, y_left, y_right = \
        None, None, None, None, None, None, None

    for feature_idx in range(n_features):
        order = X[:, feature_idx].argsort()
        X_sorted = X[order]
        y_sorted = y[order]

        for sample_idx in range(1, n_samples):
            y_left_, y_right_ = np.split(y_sorted, [sample_idx])
            gini = groups_gini_index(y_left_, y_right_)

            if best_gini is None or gini < best_gini:
                best_gini = gini
                split_feature = feature_idx
                split_value = (X_sorted[sample_idx, feature_idx] + X_sorted[
                    sample_idx - 1, feature_idx]) / 2
                y_left, y_right = y_left_, y_right_
                X_left, X_right = np.split(X_sorted, [sample_idx])

    return split_feature, split_value, X_left, X_right, y_left, y_right


def create_bags(X: np.array, y: np.array, n_bags: int, fraction: float
                ) -> List[Tuple[np.array, np.array]]:
    n_samples = X.shape[0]
    bags = []
    for _ in range(n_bags):
        indexes = np.random.choice(n_samples, int(n_samples * fraction),
                                   replace=True)
        bags.append((X[indexes], y[indexes]))
    return bags


def get_most_frequent_element_value(predictions: np.array) -> Union[int, float]:
    values, counts = np.unique(predictions, return_counts=True)
    idx = np.argmax(counts)
    return values[idx]
