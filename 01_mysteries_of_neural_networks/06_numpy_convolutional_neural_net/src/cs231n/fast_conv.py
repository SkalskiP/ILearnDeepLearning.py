# ------------------------------------------------------------------------------
# File was originally part of Stamford CS231N course: https://cs231n.github.io/
# ------------------------------------------------------------------------------

from typing import Tuple

import numpy as np


def get_im2col_idx(
    array_shape: Tuple[int, int, int, int],
    filter_dim: Tuple[int, int] = (3, 3),
    pad: int = 0,
    stride: int = 1
) -> Tuple[np.array, np.array, np.array]:
    """
    :param array_shape - 4 element tuple (n, c, h_in, w_in)
    :param filter_dim - 2 element tuple (h_f, w_f)
    :param pad - padding along width and height of input volume
    :param stride - stride along width and height of input volume
    :output 2D tensor
    ------------------------------------------------------------------------
    n - number of examples in batch
    w_in - width of input volume
    h_in - width of input volume
    c - number of channels of the input volume
    """
    n, c, h_in, w_in = array_shape
    h_f, w_f = filter_dim

    h_out = (h_in + 2 * pad - h_f) // stride + 1
    w_out = (w_in + 2 * pad - w_f) // stride + 1

    i0 = np.repeat(np.arange(h_f), w_f)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(h_out), w_out)
    j0 = np.tile(np.arange(w_f), h_f * c)
    j1 = stride * np.tile(np.arange(w_out), h_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), h_f * w_f).reshape(-1, 1)
    return k, i, j


def im2col(
    array: np.array,
    filter_dim: Tuple[int, int] = (3, 3),
    pad: int = 0,
    stride: int = 1
) -> np.array:
    """
    :param array - 4D tensor with shape (n, c, h_in, w_in)
    :param filter_dim - 2 element tuple (h_f, w_f)
    :param pad - padding along width and height of input volume
    :param stride - stride along width and height of input volume
    :output 2D tensor with shape (h_f * w_f * c, h_out, w_out * n)
    ------------------------------------------------------------------------
    n - number of examples in batch
    w_in - width of input volume
    h_in - width of input volume
    w_in - width of output volume
    h_in - width of output volume
    w_f - width of filter volume
    h_f - height of filter volume
    c - number of channels of the input volume
    """
    _, c, _, _ = array.shape
    h_f, w_f = filter_dim
    array_pad = np.pad(
        array=array,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant'
    )
    k, i, j = get_im2col_idx(
        array_shape=array.shape,
        filter_dim=filter_dim,
        pad=pad,
        stride=stride
    )
    cols = array_pad[:, k, i, j]
    return cols.transpose(1, 2, 0).reshape(h_f * w_f * c, -1)


def col2im(
    cols: np.array,
    array_shape: Tuple[int, int, int, int],
    filter_dim: Tuple[int, int] = (3, 3),
    pad: int = 0,
    stride: int = 1
) -> np.array:
    """
    :param cols - 2D tensor with shape (h_f * w_f * c, h_out, w_out * n)
    :param array_shape - 4 element tuple (n, c, h_in, w_in)
    :param filter_dim - 2 element tuple (h_f, w_f)
    :param pad - padding along width and height of input volume
    :param stride - stride along width and height of input volume
    :output 4D tensor with shape (n, c, h_in, w_in)
    ------------------------------------------------------------------------
    n - number of examples in batch
    w_in - width of input volume
    h_in - width of input volume
    w_in - width of output volume
    h_in - width of output volume
    w_f - width of filter volume
    h_f - height of filter volume
    c - number of channels of the input volume
    """
    n, c, h_in, w_in = array_shape
    h_f, w_f = filter_dim
    h_pad, w_pad = h_in + 2 * pad, w_in + 2 * pad
    array_pad = np.zeros((n, c, h_pad, w_pad), dtype=cols.dtype)
    k, i, j = get_im2col_idx(
        array_shape=array_shape,
        filter_dim=filter_dim,
        pad=pad,
        stride=stride
    )
    cols_reshaped = cols.reshape(c * h_f * w_f, -1, n)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(array_pad, (slice(None), k, i, j), cols_reshaped)
    return array_pad[:, :, pad:pad+h_in, pad:pad+w_in]
