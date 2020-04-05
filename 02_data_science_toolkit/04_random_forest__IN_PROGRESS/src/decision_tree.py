import numpy as np

from src.utils import find_split


class Node:
    def __init__(
        self, assigned_label: int = None,
        split_feature: int = None,
        split_value: float = None
    ) -> None:
        self.assigned_label = assigned_label
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = None
        self.right_child = None


class TreeClassifier:
    def __init__(self, max_depth: int, min_samples_split: int) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = Node()

    def fit(self, X: np.array, y: np.array) -> None:
        self.__split(X, y, self.root, 1)

    def predict(self, X: np.array) -> np.array:
        predictions = np.empty(X.shape[0])
        for idx, x in enumerate(X):
            predictions[idx] = self.__single_example_prediction(x)
        return predictions

    def __split(self, X: np.array, y: np.array, node: Node, depth: int) -> None:
        n_samples = y.shape[0]

        if n_samples <= self.min_samples_split or depth >= self.max_depth:
            unique, counts = np.unique(y, return_counts=True)
            node.assigned_label = unique[np.argmax(counts)]
            return

        node.split_feature, node.split_value, X_left, X_right, y_left, y_right = \
            find_split(X, y)
        node.left_child, node.right_child = Node(), Node()

        self.__split(X_left, y_left, node.left_child, depth + 1)
        self.__split(X_right, y_right, node.right_child, depth + 1)

    def __single_example_prediction(self, x: np.array) -> int:
        node = self.root
        while node.assigned_label is None:
            if x[node.split_feature] > node.split_value:
                node = node.right_child
            else:
                node = node.left_child
        return node.assigned_label
