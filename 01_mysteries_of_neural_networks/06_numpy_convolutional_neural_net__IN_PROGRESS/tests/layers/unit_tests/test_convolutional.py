import numpy as np
import pytest

from src.errors import InvalidPaddingModeError
from src.layers.convolutional import ConvLayer2D


class TestConvLayer2D:

    def test_single_convolution_step_one_channel(self):
        # given
        activation = np.array([
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0]]
        ])

        filter_W = np.array([
            [[9.0], [8.0], [7.0]],
            [[6.0], [5.0], [4.0]],
            [[3.0], [2.0], [1.0]]
        ])

        filter_b = 5.0

        expected_result = (
            2 * 1.0 * 9.0 +
            2 * 2.0 * 8.0 +
            2 * 3.0 * 7.0 +
            2 * 4.0 * 6.0 +
            5.0 * 5.0 + 5.0)

        # when
        result = ConvLayer2D.single_convolution_step(
            activation_slice=activation,
            filter_W=filter_W,
            filter_b=filter_b
        )

        # then
        assert result == expected_result

    def test_single_convolution_step_tree_channels(self):
        # given
        activation = np.array([
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
            [[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]]
        ])

        filter_W = np.array([
            [[9.0, 9.0, 9.0], [8.0, 8.0, 8.0], [7.0, 7.0, 7.0]],
            [[6.0, 6.0, 6.0], [5.0, 5.0, 5.0], [4.0, 4.0, 4.0]],
            [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]
        ])

        filter_b = 5.0

        expected_result = (
            2 * 1.0 * 9.0 +
            2 * 2.0 * 8.0 +
            2 * 3.0 * 7.0 +
            2 * 4.0 * 6.0 +
            5.0 * 5.0) * 3 + 5.0

        # when
        result = ConvLayer2D.single_convolution_step(
            activation_slice=activation,
            filter_W=filter_W,
            filter_b=filter_b
        )

        # then
        assert result == expected_result

    def test_pad_activation_valid_padding(self):
        # given
        activation = np.array([
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
            [[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]]
        ])

        # when
        result_activation = ConvLayer2D.pad_activation(
            activation=activation,
            fd=3,
            mode='valid'
        )

        # then
        assert np.alltrue(activation == result_activation)

    def test_pad_activation_same_padding(self):
        # given
        activation = np.array([
            [[[1.], [1.], [1.]], [[2.], [2.], [2.]], [[3.], [3.], [3.]]],
            [[[4.], [4.], [4.]], [[5.], [5.], [5.]], [[6.], [6.], [6.]]],
            [[[7.], [7.], [7.]], [[8.], [8.], [8.]], [[9.], [9.], [9.]]]
        ])

        expected_activation = np.array([
            [
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]]
            ],
            [
                [[0.], [0.], [0.]],
                [[1.], [1.], [1.]],
                [[2.], [2.], [2.]],
                [[3.], [3.], [3.]],
                [[0.], [0.], [0.]]],
            [
                [[0.], [0.], [0.]],
                [[4.], [4.], [4.]],
                [[5.], [5.], [5.]],
                [[6.], [6.], [6.]],
                [[0.], [0.], [0.]]
            ],
            [
                [[0.], [0.], [0.]],
                [[7.], [7.], [7.]],
                [[8.], [8.], [8.]],
                [[9.], [9.], [9.]],
                [[0.], [0.], [0.]]
            ],
            [
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]]
            ]
        ])

        # when
        result_activation = ConvLayer2D.pad_activation(
            activation=activation,
            fd=3,
            mode='same'
        )

        # then
        assert np.alltrue(expected_activation == result_activation)
        assert result_activation.shape == (5, 5, 3, 1)

    def test_pad_activation_invalid_padding_mode(self):
        # given
        activation = np.array([
            [[[1.], [1.], [1.]], [[2.], [2.], [2.]], [[3.], [3.], [3.]]],
            [[[4.], [4.], [4.]], [[5.], [5.], [5.]], [[6.], [6.], [6.]]],
            [[[7.], [7.], [7.]], [[8.], [8.], [8.]], [[9.], [9.], [9.]]]
        ])

        # when
        with pytest.raises(InvalidPaddingModeError):
            _ = ConvLayer2D.pad_activation(
                activation=activation,
                fd=3,
                mode='lorem ipsum'
            )

    def test_calculate_forward_pass_output_shape_valid_padding(self):
        # given
        activation = np.random.rand(11, 11, 3, 64)
        W = np.random.rand(64, 5, 5, 3)

        # when
        result = ConvLayer2D.calculate_forward_pass_output_shape(
            activation=activation,
            W=W,
            mode="valid"
        )

        # then
        assert result == (7, 7, 64, 64)

    def test_calculate_forward_pass_output_shape_same_padding(self):
        # given
        activation = np.random.rand(11, 11, 3, 64)
        W = np.random.rand(64, 5, 5, 3)

        # when
        result = ConvLayer2D.calculate_forward_pass_output_shape(
            activation=activation,
            W=W,
            mode="same"
        )

        # then
        assert result == (11, 11, 64, 64)

    def test_calculate_forward_pass_output_shape_invalid_padding_mode(self):
        # given
        activation = np.random.rand(11, 11, 3, 64)
        W = np.random.rand(64, 5, 5, 3)

        # when
        with pytest.raises(InvalidPaddingModeError):
            _ = ConvLayer2D.calculate_forward_pass_output_shape(
                activation=activation,
                W=W,
                mode="lorem ipsum"
            )

    def test_forward_pass_only_size_same_padding(self):
        # given
        activation = np.random.rand(11, 11, 3, 64)
        W = np.random.rand(16, 5, 5, 3)
        b = np.random.rand(16)
        layer = ConvLayer2D(W=W, b=b, padding='same')

        # when
        result = layer.forward_pass(activation)

        # then
        assert result.shape == (11, 11, 16, 64)

    def test_forward_pass_only_size_valid_padding(self):
        # given
        activation = np.random.rand(28, 28, 1, 64)
        W = np.random.rand(16, 3, 3, 1)
        b = np.random.rand(16)
        layer = ConvLayer2D(W=W, b=b, padding='valid')

        # when
        result = layer.forward_pass(activation)

        # then
        assert result.shape == (26, 26, 16, 64)

    def test_backward_pass_only_size_same_padding(self):
        # given
        activation = np.random.rand(11, 11, 3, 64)
        W = np.random.rand(16, 5, 5, 3)
        b = np.random.rand(16)
        layer = ConvLayer2D(W=W, b=b, padding='same')

        # when
        forward_result = layer.forward_pass(activation)
        backward_result = layer.backward_pass(forward_result)

        # then
        assert backward_result.shape == activation.shape

    def test_backward_pass_only_size_valid_padding(self):
        # given
        activation = np.random.rand(11, 11, 3, 64)
        W = np.random.rand(16, 5, 5, 3)
        b = np.random.rand(16)
        layer = ConvLayer2D(W=W, b=b, padding='valid')

        # when
        forward_result = layer.forward_pass(activation)
        backward_result = layer.backward_pass(forward_result)

        # then
        assert backward_result.shape == activation.shape





