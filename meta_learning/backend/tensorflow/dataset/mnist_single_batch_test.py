

import unittest
import numpy as _np
import tensorflow as tf
from tf_utils import debug as _debug

from meta_learning.backend.tensorflow.dataset import mnist_single_batch as _mnist_single_batch

DATA_DIR = '/tmp/mnist_test'


class TestMnistSingleBatch(unittest.TestCase):

    def _test_iterator(self, batch_size):
        data_dir = DATA_DIR
        num_epochs = 1
        for shard_name in ['train',
                           'validation',
                           'test']:
            # Note we need a new graph otherwise we always connect to the
            # same graph and redefine the same variables.
            tf.reset_default_graph()
            with tf.Session() as sess:
                mnist = _mnist_single_batch.MnistSingleBatch(
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    shard_name=shard_name,
                    data_dir=data_dir)
                input_fn = mnist.create_input_fn()
                result = input_fn()
                features, labels = sess.run(result)

                self.assertEqual(mnist.num_examples, batch_size)

                self.assertTrue(_np.array_equal(features['images'].shape,
                                                (batch_size, 28, 28, 1)))
                self.assertTrue(_np.array_equal(labels['labels'].shape,
                                                (batch_size, 10)))
                new_features, new_labels = sess.run(result)
                for key in new_features.keys():
                    self.assertTrue(_np.array_equal(features[key],
                                                    new_features[key]))
                for key in new_labels.keys():
                    self.assertTrue(_np.array_equal(labels[key],
                                                    new_labels[key]))

    def test_iterator(self):
        for batch_size in [1, 2, 8, 16]:
            try:
                self._test_iterator(batch_size)
            except BaseException:
                _debug.ipdb_exception()


if __name__ == '__main__':
    unittest.main()
