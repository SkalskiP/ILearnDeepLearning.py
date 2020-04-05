import numpy as np

from src.decision_tree import TreeClassifier
from src.utils import get_most_frequent_element_value, create_bags


class RandomForestClassifier:
    def __init__(self, n_estimators: int, fraction: float, max_depth: int,
                 min_samples_split: int):
        self.n_estimators = n_estimators
        self.fraction = fraction
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.estimators = [
            TreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            for _ in range(self.n_estimators)
        ]

    def fit(self, X: np.array, y: np.array) -> None:
        bags = create_bags(X, y, self.n_estimators, self.fraction)
        for estimator, (X_bag, y_bag) in zip(self.estimators, bags):
            estimator.fit(X_bag, y_bag)

    def predict(self, X: np.array) -> np.array:
        n_samples = X.shape[0]
        predictions_raw = np.empty((self.n_estimators, n_samples))
        predictions = np.empty(n_samples)

        for i in range(self.n_estimators):
            pred = self.estimators[i].predict(X)
            predictions_raw[i] = pred

        for i in range(n_samples):
            predictions[i] = get_most_frequent_element_value(predictions_raw[:, i])
        return predictions
