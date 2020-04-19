import numpy as np

from src.utils.core import convert_prob2one_hot, convert_categorical2one_hot


class TestConvertProb2OneHot:

    def test_conversion_with_2_classes(self):
        # given
        prob = np.array([
            [0.1, 0.7, 0.8, 0.2],
            [0.9, 0.3, 0.2, 0.8]
        ])
        expected_one_hot = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])

        # when
        result_one_hot = convert_prob2one_hot(prob)

        # then
        assert np.alltrue(result_one_hot == expected_one_hot)

    def test_conversion_with_3_classes(self):
        # given
        prob = np.array([
            [0.3, 0.7, 0.8, 0.2],
            [0.3, 0.1, 0.1, 0.8],
            [0.4, 0.2, 0.1, 0.0]
        ])
        expected_one_hot = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ])

        # when
        result_one_hot = convert_prob2one_hot(prob)

        # then
        assert np.alltrue(result_one_hot == expected_one_hot)


class TestConvertCategoricalToOneHot:

    def test_conversion_with_2_classes(self):
        # given
        categories = np.array([0, 1, 1, 0, 1])
        expected_one_hot = np.array([
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1]
        ])

        # when
        result_one_hot = convert_categorical2one_hot(categories)

        # then
        assert np.alltrue(result_one_hot == expected_one_hot)

    def test_conversion_with_3_classes(self):
        # given
        categories = np.array([0, 2, 1, 0, 1])
        expected_one_hot = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])

        # when
        result_one_hot = convert_categorical2one_hot(categories)

        # then
        assert np.alltrue(result_one_hot == expected_one_hot)
