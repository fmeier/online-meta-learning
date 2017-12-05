"""Interface for data provider we use for learning.
"""

import numpy as np

import tensorflow as tf

from tf_utils.dataset import mnist as _mnist
from meta_learning.backend.tensorflow.dataset import interface as _interface


class MnistClassPair(_mnist.Mnist):
    """Mnist but only examples from a single class.
    """

    @classmethod
    def create_from_params(cls, params, *args, **kwargs):
        return cls(params.num_epochs,
                   params.batch_size,
                   params.shard_name,
                   params.shuffle,
                   params.data_dir,
                   params.class_positive,
                   params.class_negative,
                   params.data_size)

    def __init__(self,
                 num_epochs,
                 batch_size,
                 shard_name,
                 shuffle,
                 data_dir,
                 class_positive,
                 class_negative,
                 data_size):
        super(MnistClassPair, self).__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            shard_name=shard_name,
            shuffle=shuffle,
            data_dir=data_dir)

        # Data size is used to default to single batch behavior.
        self._data_size = data_size
        self._class_positive = class_positive
        self._class_negative = class_negative

        assert self._class_positive in range(10), (
            'class_positive has to be between 0 and 9')
        assert self._class_negative in range(10), (
            'class_negative has to be between 0 and 9')

    @property
    def num_examples(self):
        if self._shard_name == _interface.SHARD_NAME_TRAIN:
            return min(self._dataset.train.num_examples, self._data_size)
        if self._shard_name == _interface.SHARD_NAME_VALIDATION:
            return min(self._dataset.validation.num_examples, self._data_size)
        if self._shard_name == _interface.SHARD_NAME_TEST:
            return min(self._dataset.test.num_examples, self._data_size)

    def _create_input_fn(self, dataset):
        if self._data_size is None:
            self._data_size = dataset.num_examples

        assert self._batch_size <= self._data_size, (
            'The batch size has to be smaller/equal than the data size.')

        # We load all data since we have to subselect.
        images, labels = dataset.next_batch(dataset.num_examples)

        positive_indices = set(
            np.where(labels[:, self._class_positive] == 1)[0].tolist())
        negative_indices = set(
            np.where(labels[:, self._class_negative] == 1)[0].tolist())

        indices = sorted(list(positive_indices.union(negative_indices)))

        # Data sub selection to images containing positive and negative
        # classes.
        images = images[indices]

        # For the labels we have to do it in two steps and stack the result.
        labels_positive = labels[indices, self._class_positive]
        labels_negative = labels[indices, self._class_negative]
        labels = np.stack(
            [labels_positive, labels_negative], axis=1)

        if self._data_size == self._batch_size:
            # This means we do not mean to shuffle, we are in single batch
            # mode. This means we can simply return the constants.
            def input_fn():
                feature_dict = {'images': tf.constant(images[:self._data_size],
                                                      dtype=tf.float32)}
                label_dict = {'labels': tf.constant(labels[:self._data_size],
                                                    dtype=tf.float32)}
                return feature_dict, label_dict
            return input_fn

        def input_fn():
            data = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
            data = data.repeat(self._num_epochs)
            if self._shuffle:
                data = data.shuffle(buffer_size=self._batch_size * 100)
            data = data.batch(self._batch_size)
            iterator = data.make_one_shot_iterator()
            # We have to change the image names due to the closure scope.
            _images, _labels = iterator.get_next()
            return {'images': _images}, {'labels': _labels}
        return input_fn
