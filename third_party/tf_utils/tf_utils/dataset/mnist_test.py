

import unittest
import numpy as _np
import tensorflow as tf
from tf_utils import debug as _debug

from tf_utils.dataset import mnist as _mnist

DATA_DIR = '/tmp/mnist_test'


class TestMnist(unittest.TestCase):

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
                mnist = _mnist.Mnist(batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     shard_name=shard_name,
                                     data_dir=data_dir,
                                     shuffle=True)
                input_fn = mnist.create_input_fn()
                result = input_fn()
                features, labels = sess.run(result)

                if shard_name == 'train':
                    self.assertEqual(mnist.num_examples, 55000)
                if shard_name == 'validation':
                    self.assertEqual(mnist.num_examples, 5000)
                if shard_name == 'test':
                    self.assertEqual(mnist.num_examples, 10000)

                self.assertTrue(_np.array_equal(features['images'].shape,
                                                (batch_size, 28, 28, 1)))
                self.assertTrue(_np.array_equal(labels['labels'].shape,
                                                (batch_size, 10)))

                new_features, new_labels = sess.run(result)
                for key in new_features.keys():
                    self.assertFalse(_np.array_equal(features[key],
                                                     new_features[key]))
                for key in new_labels.keys():
                    self.assertFalse(_np.array_equal(labels[key],
                                                     new_labels[key]))

    def test_iterator(self):
        for batch_size in [1, 2, 8, 16]:
            try:
                self._test_iterator(batch_size)
            except:
                _debug.ipdb_exception()


if __name__ == '__main__':
    unittest.main()
