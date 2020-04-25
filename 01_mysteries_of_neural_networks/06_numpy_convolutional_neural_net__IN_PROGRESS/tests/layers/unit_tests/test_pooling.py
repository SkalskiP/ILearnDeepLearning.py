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

    def test_backward_pass_single_channel_single_item(self):
        # given
        pool_size = (2, 2)
        strides = 2
        forward_activation = np.array([
            [[[1]], [[2]], [[2]], [[1]]],
            [[[3]], [[4]], [[0]], [[0]]],
            [[[5]], [[2]], [[1]], [[1]]],
            [[[3]], [[4]], [[0]], [[3]]]
        ])

        backward_activation = np.array([
            [[[3]], [[1]]],
            [[[8]], [[2]]],
        ])

        expected_backward_result = np.array([
            [[[0]], [[0]], [[1]], [[0]]],
            [[[0]], [[3]], [[0]], [[0]]],
            [[[8]], [[0]], [[0]], [[0]]],
            [[[0]], [[0]], [[0]], [[2]]]
        ])

        # when
        layer = MaxPoolLayer(pool_size=pool_size, strides=strides)
        _ = layer.forward_pass(forward_activation)
        backward_result = layer.backward_pass(backward_activation)

        # then
        assert np.alltrue(expected_backward_result == backward_result)

    def test_backward_pass_two_channels_single_item(self):
        # given
        pool_size = (2, 2)
        strides = 2
        forward_activation = np.array([
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

        backward_activation = np.array([
            [
                [[7], [2]],
                [[4], [3]]
            ],
            [
                [[1], [5]],
                [[2], [2]]
            ]
        ])

        expected_backward_result = np.array([
            [
                [[0], [2]],
                [[0], [0]],
                [[4], [0]],
                [[0], [0]]
            ],
            [
                [[0], [0]],
                [[7], [0]],
                [[0], [3]],
                [[0], [0]]
            ],
            [
                [[1], [0]],
                [[0], [0]],
                [[0], [0]],
                [[0], [0]]
            ],
            [
                [[0], [0]],
                [[0], [5]],
                [[0], [2]],
                [[2], [0]]
            ]
        ])

        # when
        layer = MaxPoolLayer(pool_size=pool_size, strides=strides)
        _ = layer.forward_pass(forward_activation)
        backward_result = layer.backward_pass(backward_activation)

        # then
        assert np.alltrue(expected_backward_result == backward_result)

    def test_backward_pass_single_channel_two_items(self):
        # given
        pool_size = (2, 2)
        strides = 2
        forward_activation = np.array([
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

        backward_activation = np.array([
            [
                [[7, 2]],
                [[4, 3]]
            ],
            [
                [[1, 5]],
                [[2, 2]]
            ]
        ])

        expected_backward_result = np.array([
            [
                [[0, 2]],
                [[0, 0]],
                [[4, 0]],
                [[0, 0]]
            ],
            [
                [[0, 0]],
                [[7, 0]],
                [[0, 3]],
                [[0, 0]]
            ],
            [
                [[1, 0]],
                [[0, 0]],
                [[0, 0]],
                [[0, 0]]
            ],
            [
                [[0, 0]],
                [[0, 5]],
                [[0, 2]],
                [[2, 0]]
            ]
        ])

        # when
        layer = MaxPoolLayer(pool_size=pool_size, strides=strides)
        _ = layer.forward_pass(forward_activation)
        backward_result = layer.backward_pass(backward_activation)

        # then
        assert np.alltrue(expected_backward_result == backward_result)