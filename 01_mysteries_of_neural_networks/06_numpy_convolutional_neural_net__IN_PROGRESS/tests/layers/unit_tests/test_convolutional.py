import numpy as np
import pytest

from src.errors import InvalidPaddingModeError
from src.layers.convolutional import ConvLayer2D


class TestConvLayer2D:

    def test_pad_symmetrical(self):
        # given
        array = np.random.rand(100, 28, 28, 3)
        pad = 3, 3

        # when
        result = ConvLayer2D.pad(array=array, pad=pad)

        # then
        print(result.sum())
        print(array.sum())
        assert result.shape == (100, 34, 34, 3)
        assert abs(result.sum() - array.sum()) < 1e-8

    def test_pad_asymmetrical(self):
        # given
        array = np.random.rand(100, 28, 28, 3)
        pad = 3, 5

        # when
        result = ConvLayer2D.pad(array=array, pad=pad)

        # then
        print(result.sum())
        print(array.sum())
        assert result.shape == (100, 34, 38, 3)
        assert abs(result.sum() - array.sum()) < 1e-8

    def test_calculate_pad_width_with_valid_padding(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        padding = 'valid'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.calculate_pad_dims()

        # then
        assert result == (0, 0)

    def test_calculate_pad_width_with_same_padding_symmetrical(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        padding = 'same'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.calculate_pad_dims()

        # then
        assert result == (2, 2)

    def test_calculate_pad_width_with_same_padding_asymmetrical(self):
        # given
        w = np.random.rand(5, 7, 3, 16)
        b = np.random.rand(16)
        padding = 'same'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.calculate_pad_dims()

        # then
        assert result == (2, 3)

    def test_calculate_pad_width_with_invalid_padding_value(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        padding = 'lorem ipsum'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        with pytest.raises(InvalidPaddingModeError):
            _ = layer.calculate_pad_dims()

    def test_calculate_output_dims_with_same_padding_symmetrical(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        padding = 'same'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.calculate_output_dims((32, 11, 11, 3))

        # then
        assert result == (32, 11, 11, 16)

    def test_calculate_output_dims_with_same_padding_asymmetrical(self):
        # given
        w = np.random.rand(3, 5, 3, 16)
        b = np.random.rand(16)
        padding = 'same'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.calculate_output_dims((32, 11, 11, 3))

        # then
        assert result == (32, 11, 11, 16)

    def test_calculate_output_dims_with_valid_padding_symmetrical(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        padding = 'valid'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.calculate_output_dims((32, 11, 11, 3))

        # then
        assert result == (32, 7, 7, 16)

    def test_calculate_output_dims_with_valid_padding_asymmetrical(self):
        # given
        w = np.random.rand(3, 5, 3, 16)
        b = np.random.rand(16)
        padding = 'valid'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.calculate_output_dims((32, 11, 11, 3))

        # then
        assert result == (32, 9, 7, 16)

    def test_calculate_output_dims_with_invalid_padding_value(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        padding = 'lorem ipsum'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        with pytest.raises(InvalidPaddingModeError):
            _ = layer.calculate_output_dims((32, 11, 11, 3))

    def test_forward_pass_with_same_padding(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        activation = np.random.rand(16, 11, 11, 3)
        padding = 'same'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.forward_pass(activation)

        assert result.shape == (16, 11, 11, 16)
        expected_val = np.sum(w[:, :, :, 0] * activation[0, 0:5, 0:5, :]) + b[0]
        assert abs(expected_val - result[0, 2, 2, 0]) < 1e-8

    def test_forward_pass_with_valid_padding(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        activation = np.random.rand(16, 11, 11, 3)
        padding = 'valid'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        result = layer.forward_pass(activation)

        assert result.shape == (16, 7, 7, 16)
        expected_val = np.sum(w[:, :, :, 0] * activation[0, 0:5, 0:5, :]) + b[0]
        assert abs(expected_val - result[0, 0, 0, 0]) < 1e-8

    def test_forward_pass_with_invalid_padding_value(self):
        # given
        w = np.random.rand(5, 5, 3, 16)
        b = np.random.rand(16)
        activation = np.random.rand(16, 11, 11, 3)
        padding = 'lorem ipsum'

        # when
        layer = ConvLayer2D(w=w, b=b, padding=padding)
        with pytest.raises(InvalidPaddingModeError):
            _ = layer.forward_pass(activation)

    # def test_backward_pass_only_size_same_padding(self):
    #     # given
    #     activation = np.random.rand(64, 11, 11, 3)
    #     w = np.random.rand(5, 5, 3, 16)
    #     b = np.random.rand(16)
    #     layer = ConvLayer2D(w=w, b=b, padding='same')
    #
    #     # when
    #     forward_result = layer.forward_pass(activation)
    #     backward_result = layer.backward_pass(forward_result)
    #
    #     # then
    #     assert backward_result.shape == activation.shape
    #
    # def test_backward_pass_only_size_valid_padding(self):
    #     # given
    #     activation = np.random.rand(64, 11, 11, 3)
    #     w = np.random.rand(5, 5, 3, 16)
    #     b = np.random.rand(16)
    #     layer = ConvLayer2D(w=w, b=b, padding='valid')
    #
    #     # when
    #     forward_result = layer.forward_pass(activation)
    #     backward_result = layer.backward_pass(forward_result)
    #
    #     # then
    #     assert backward_result.shape == activation.shape
