import numpy as np

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
            activation=activation,
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
            activation=activation,
            filter_W=filter_W,
            filter_b=filter_b
        )

        # then
        assert result == expected_result
