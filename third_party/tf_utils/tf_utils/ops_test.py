"""Tests for ops.
"""
import unittest

import numpy as np
import tensorflow as tf

from tf_utils import utils
from tf_utils import ops


class TestOps(unittest.TestCase):

    def test_concat_with_shape(self):
        data = np.array([[0.5, 0.5, 0.0, 0.0, 0.6, 0.0],
                         [0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.7, 0.7, 0.7]],
                        dtype=np.float32)

        with utils.device_cpu():
            with tf.Session():
                data_0 = tf.constant(data)
                data_1 = tf.constant(data)
                concat_0 = ops.concat_with_shape(
                    0, [data_0, data_1])
                concat_1 = ops.concat_with_shape(
                    1, [data_0, data_1])
                concat_0_np = concat_0.eval()
                concat_1_np = concat_1.eval()
                self.assertTrue(np.array_equal(concat_0_np.shape,
                                               [14, 6]))
                self.assertTrue(np.array_equal(concat_1_np.shape,
                                               [7, 12]))

    def test_mask_one_row(self):
        data = np.array([[0.1, 0.1, 0.1],
                         [0.1, 0.05, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.1, 0.2, 0.3]], dtype=np.float32)
        ref = np.array([[0.1, 0.0, 0.0],
                        [0.1, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.3]], dtype=np.float32)
        with utils.device_cpu():
            with tf.Session():
                res = ops.mask_argmax_row(tf.constant(data))
                self.assertTrue(np.array_equal(res.eval(), ref))


if __name__ == '__main__':
    unittest.main()
