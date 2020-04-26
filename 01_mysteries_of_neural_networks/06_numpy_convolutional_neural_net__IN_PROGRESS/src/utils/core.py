import numpy as np


def convert_categorical2one_hot(y: np.array) -> np.array:
    """
    k - number of classes
    N - number of items
    :param y - categorical array with (N, 1) shape
    :return one hot array with (N, k) shape
    """
    one_hot_matrix = np.zeros((y.size, y.max() + 1))
    one_hot_matrix[np.arange(y.size), y] = 1
    return one_hot_matrix


def convert_prob2categorical(probs: np.array) -> np.array:
    """
    k - number of classes
    N - number of items
    :param probs - softmax output array with (N, k) shape
    :return categorical array with (N, ) shape
    """
    return np.argmax(probs, axis=1)


def convert_prob2one_hot(probs: np.array) -> np.array:
    """
    k - number of classes
    N - number of items
    :param probs - softmax output array with (N, k) shape
    :return one hot array with (N, k) shape
    """
    class_idx = convert_prob2categorical(probs)
    one_hot_matrix = np.zeros_like(probs)
    one_hot_matrix[np.arange(probs.shape[0]), class_idx] = 1
    return one_hot_matrix


def generate_batches(x: np.array, y: np.array, batch_size: int):
    """
    N - number of examples in data set
    n - number of examples in batch [batch_size]
    k - number of classes
    :param x - features array with (N, ...) shape
    :param y - one hot ground truth array with (N, k) shape
    :batch_size - number of elements in single batch
    """
    for i in range(0, x.shape[-1], batch_size):
        yield (
            x.take(indices=range(
                i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(
                i, min(i + batch_size, y.shape[0])), axis=0)
        )

