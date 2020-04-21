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
    :param probs - softmax output array with (k, N) shape
    :return categorical array with (N, ) shape
    """
    return np.argmax(probs, axis=0)


def convert_prob2one_hot(probs: np.array) -> np.array:
    """
    k - number of classes
    N - number of items
    :param probs - softmax output array with (k, N) shape
    :return one hot array with (k, N) shape
    """
    class_idx = convert_prob2categorical(probs)
    one_hot_matrix = np.zeros(probs.shape)
    one_hot_matrix[class_idx, np.arange(probs.shape[1])] = 1
    return one_hot_matrix


def generate_batches(x: np.array, y: np.array, batch_size: int):
    """
    k - number of classes
    N - number of items
    :param x - features array with (..., N) shape
    :param y - one hot ground truth array with (k, N) shape
    :batch_size - number of elements in single batch
    """
    for i in range(0, x.shape[-1], batch_size):
        yield (
            x.take(indices=range(
                i, min(i + batch_size, x.shape[-1])), axis=len(x.shape) - 1),
            y.take(indices=range(
                i, min(i + batch_size, y.shape[-1])), axis=len(y.shape) - 1)
        )

