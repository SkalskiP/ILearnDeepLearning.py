import numpy as np

from src.layers.pooling import MaxPoolLayer


class TestMaxPoolLayer:

    def test_forward_pass_single_channel_single_item(self):
        # given
        pool_size = (2, 2)
        strides = 2
        activation = np.array([
            [[[1]], [[2]], [[2]], [[1]]],
            [[[3]], [[4]], [[0]], [[0]]],
            [[[5]], [[2]], [[1]], [[1]]],
            [[[3]], [[4]], [[0]], [[3]]]
        ])

        expected_result = np.array([
            [[[4]], [[2]]],
            [[[5]], [[3]]],
        ])

        # when
        layer = MaxPoolLayer(pool_size=pool_size, strides=strides)
        result = layer.forward_pass(activation)

        # then
        assert np.alltrue(expected_result == result)

    def test_forward_pass_two_channels_single_item(self):
        # given
        pool_size = (2, 2)
        strides = 2
        activation = np.array([
            [
                [[1], [5]],
                [[2], [2]],
                [[2], [2]],
                [[1], [1]]
            ],
            [
                [[3], [3]],
                [[4], [4]],
                [[0], [3]],
                [[0], [0]]
            ],
            [
                [[5], [2]],
                [[2], [2]],
                [[1], [1]],
                [[1], [1]]
            ],
            [
                [[3], [3]],
                [[4], [4]],
                [[0], [2]],
                [[3], [0]]
            ]
        ])

        expected_result = np.array([
            [
                [[4], [5]],
                [[2], [3]]
            ],
            [
                [[5], [4]],
                [[3], [2]]
            ]
        ])

        # when
        layer = MaxPoolLayer(pool_size=pool_size, strides=strides)
        result = layer.forward_pass(activation)

        # then
        assert np.alltrue(expected_result == result)

    def test_forward_pass_single_channel_two_items(self):
        # given
        pool_size = (2, 2)
        strides = 2
        activation = np.array([
            [
                [[1, 5]],
                [[2, 2]],
                [[2, 2]],
                [[1, 1]]
            ],
            [
                [[3, 3]],
                [[4, 4]],
                [[0, 3]],
                [[0, 0]]
            ],
            [
                [[5, 2]],
                [[2, 2]],
                [[1, 1]],
                [[1, 1]]
            ],
            [
                [[3, 3]],
                [[4, 4]],
                [[0, 2]],
                [[3, 0]]
            ]
        ])

        expected_result = np.array([
            [
                [[4, 5]],
                [[2, 3]]
            ],
            [
                [[5, 4]],
                [[3, 2]]
            ]
        ])

        # when
        layer = MaxPoolLayer(pool_size=pool_size, strides=strides)
        result = layer.forward_pass(activation)

        # then
        assert np.alltrue(expected_result == result)
