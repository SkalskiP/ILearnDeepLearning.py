import numpy as np

from src.layers.flatten import FlattenLayer


class TestFlattenLayer:

    def test_flatten_forward_pass_with_tree_axis(self):
        # given
        activation = np.random.rand(100, 28, 28)

        # when
        flatten_layer = FlattenLayer()
        result = flatten_layer.forward_pass(activation)

        # then
        assert result.shape == (100, 28 * 28)

    def test_flatten_forward_pass_with_four_axis(self):
        # given
        activation = np.random.rand(100, 28, 28, 3)

        # when
        flatten_layer = FlattenLayer()
        result = flatten_layer.forward_pass(activation)

        # then
        assert result.shape == (100, 28 * 28 * 3)

    def test_flatten_backward_pass_with_tree_axis(self):
        # given
        activation = np.random.rand(100, 28, 28)

        # when
        flatten_layer = FlattenLayer()
        forward_result = flatten_layer.forward_pass(activation)
        backward_result = flatten_layer.backward_pass(forward_result)

        # then
        assert np.alltrue(activation == backward_result)

    def test_flatten_backward_pass_with_four_axis(self):
        # given
        activation = np.random.rand(100, 28, 28, 3)

        # when
        flatten_layer = FlattenLayer()
        forward_result = flatten_layer.forward_pass(activation)
        backward_result = flatten_layer.backward_pass(forward_result)

        # then
        assert np.alltrue(activation == backward_result)
