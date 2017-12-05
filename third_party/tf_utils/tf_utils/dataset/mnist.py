
import numpy as np
import tensorflow as tf

from tf_utils.dataset import interface as _interface


class Mnist(_interface.Interface):
    """Just the standard mnist example.
    """

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(params.num_epochs,
                   params.batch_size,
                   params.shard_name,
                   params.shuffle,
                   params.data_dir)

    def __init__(self, batch_size, num_epochs, shard_name, shuffle, data_dir):
        super(Mnist, self).__init__(batch_size=batch_size,
                                    num_epochs=num_epochs,
                                    shard_name=shard_name)
        self._data_dir = data_dir
        self._dataset = tf.contrib.learn.datasets.mnist.read_data_sets(
            train_dir=self._data_dir,
            reshape=False,
            one_hot=True)
        self._shuffle = shuffle

    @property
    def num_examples(self):
        if self._shard_name == _interface.SHARD_NAME_TRAIN:
            return self._dataset.train.num_examples
        if self._shard_name == _interface.SHARD_NAME_VALIDATION:
            return self._dataset.validation.num_examples
        if self._shard_name == _interface.SHARD_NAME_TEST:
            return self._dataset.test.num_examples

    def _create_input_fn(self, dataset):
        def input_fn():
            images, labels = dataset.next_batch(self._batch_size)

            def input_fn():
                images, labels = dataset.next_batch(self._batch_size)
                return images.astype(np.float32), labels.astype(np.float32)

            tf_images, tf_labels = tf.py_func(
                input_fn, [], [tf.float32, tf.float32])
            tf_images = tf.reshape(tf_images, shape=images.shape)
            tf_labels = tf.reshape(tf_labels, shape=labels.shape)

            # As soon as there exist the dataset from generator we should
            # switch to that implementation.
            # data = tf.contrib.data.Dataset.from_tensor_slices(
            #     (tf_images, tf_labels))
            # data = data.repeat(None)
            # if self._shuffle:
            #     data = data.shuffle(buffer_size=self._batch_size * 100)
            # data = data.batch(self._batch_size)
            # iterator = data.make_one_shot_iterator()
            # images, labels = iterator.get_next()
            return {'images': tf_images}, {'labels': tf_labels}
        return input_fn

    def create_input_fn(self):
        if self._shard_name == _interface.SHARD_NAME_TRAIN:
            return self._create_input_fn(self._dataset.train)
        if self._shard_name == _interface.SHARD_NAME_VALIDATION:
            return self._create_input_fn(self._dataset.validation)
        if self._shard_name == _interface.SHARD_NAME_TEST:
            return self._create_input_fn(self._dataset.test)
