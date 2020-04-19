import numpy as np

from src.layers.flatten import FlattenLayer


class TestFlattenLayer:

    def test_flatten_forward_pass_with_tree_axis(self):
        # given
        activation = np.random.rand(28, 28, 100)

        # when
        flatten_layer = FlattenLayer()
        result = flatten_layer.forward_pass(activation)

        # then
        assert result.shape == (28 * 28, 100)

    def test_flatten_forward_pass_with_four_axis(self):
        # given
        activation = np.random.rand(28, 28, 3, 100)

        # when
        flatten_layer = FlattenLayer()
        result = flatten_layer.forward_pass(activation)

        # then
        assert result.shape == (28 * 28 * 3, 100)

    def test_flatten_backward_pass_with_tree_axis(self):
        # given
        activation = np.random.rand(28, 28, 100)

        # when
        flatten_layer = FlattenLayer()
        forward_result = flatten_layer.forward_pass(activation)
        backward_result = flatten_layer.backward_pass(forward_result)

        # then
        assert np.alltrue(activation == backward_result)

    def test_flatten_backward_pass_with_four_axis(self):
        # given
        activation = np.random.rand(28, 28, 3, 100)

        # when
        flatten_layer = FlattenLayer()
        forward_result = flatten_layer.forward_pass(activation)
        backward_result = flatten_layer.backward_pass(forward_result)

        # then
        assert np.alltrue(activation == backward_result)
