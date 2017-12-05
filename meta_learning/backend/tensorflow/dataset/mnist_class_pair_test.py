

import unittest
import numpy as _np
import tensorflow as tf
from tf_utils import debug as _debug

from meta_learning.backend.tensorflow.dataset import mnist_class_pair as _mnist_class_pair

DATA_DIR = '/tmp/mnist_test'


class TestMnistClassPair(unittest.TestCase):

    def _test_iterator(self,
                       class_positive,
                       class_negative,
                       batch_size,
                       data_size):
        data_dir = DATA_DIR
        num_epochs = 1
        for shard_name in ['train',
                           'validation',
                           'test']:
            # Note we need a new graph otherwise we always connect to the
            # same graph and redefine the same variables.
            tf.reset_default_graph()
            with tf.Session() as sess:
                mnist = _mnist_class_pair.MnistClassPair(
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    shard_name=shard_name,
                    data_dir=data_dir,
                    class_positive=class_positive,
                    class_negative=class_negative,
                    data_size=data_size,
                    shuffle=True)
                input_fn = mnist.create_input_fn()
                result = input_fn()
                features, labels = sess.run(result)

                if data_size is not None:
                    self.assertEqual(mnist.num_examples, data_size)
                else:
                    if shard_name == 'train':
                        self.assertEqual(mnist.num_examples, 55000)
                    if shard_name == 'validation':
                        self.assertEqual(mnist.num_examples, 5000)
                    if shard_name == 'test':
                        self.assertEqual(mnist.num_examples, 10000)

                _data_size = mnist.num_examples
                self.assertTrue(_np.array_equal(features['images'].shape,
                                                (batch_size, 28, 28, 1)))
                self.assertTrue(_np.array_equal(labels['labels'].shape,
                                                (batch_size, 2)))
                new_features, new_labels = sess.run(result)
                if batch_size != _data_size:
                    for key in new_features.keys():
                        self.assertFalse(_np.array_equal(features[key],
                                                         new_features[key]))
                    # Labels might not be different every time.
                else:
                    for key in new_features.keys():
                        self.assertTrue(_np.array_equal(features[key],
                                                        new_features[key]))
                    for key in new_labels.keys():
                        self.assertTrue(_np.array_equal(labels[key],
                                                        new_labels[key]))

    def test_iterator_batch(self):
        for class_pair in xrange(10):
            for batch_size in [1, 16]:
                try:
                    self._test_iterator(0, class_pair, batch_size, batch_size)
                except BaseException:
                    _debug.ipdb_exception()

    def test_iterator_full(self):
        for class_pair in xrange(10):
            for batch_size in [1, 16]:
                try:
                    self._test_iterator(class_positive=0,
                                        class_negative=class_pair,
                                        batch_size=batch_size,
                                        data_size=None)
                except BaseException:
                    _debug.ipdb_exception()


if __name__ == '__main__':
    unittest.main()
