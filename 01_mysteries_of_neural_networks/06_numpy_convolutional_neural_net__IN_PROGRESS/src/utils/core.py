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
    :param probs - softmax output array with (n, k) shape
    :return categorical array with (n, ) shape
    ----------------------------------------------------------------------------
    k - number of classes
    n - number of examples
    """
    return np.argmax(probs, axis=1)


def convert_prob2one_hot(probs: np.array) -> np.array:
    """
    :param probs - softmax output array with (n, k) shape
    :return one hot array with (n, k) shape
    ----------------------------------------------------------------------------
    k - number of classes
    n - number of examples
    """
    class_idx = convert_prob2categorical(probs)
    one_hot_matrix = np.zeros_like(probs)
    one_hot_matrix[np.arange(probs.shape[0]), class_idx] = 1
    return one_hot_matrix


def generate_batches(x: np.array, y: np.array, batch_size: int):
    """
    :param x - features array with (n, ...) shape
    :param y - one hot ground truth array with (n, k) shape
    :batch_size - number of elements in single batch
    ----------------------------------------------------------------------------
    n - number of examples in data set
    k - number of classes
    """
    for i in range(0, x.shape[0], batch_size):
        yield (
            x.take(indices=range(
                i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(
                i, min(i + batch_size, y.shape[0])), axis=0)
        )

