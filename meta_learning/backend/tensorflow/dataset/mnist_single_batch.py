"""Interface for data provider we use for learning.
"""

import tensorflow as tf

from tf_utils.dataset import mnist as _mnist

from meta_learning.backend.tensorflow.dataset import interface as _interface


class MnistSingleBatch(_mnist.Mnist):
    """Select a determinist mnist subpart to perform full dataset optimization.
    """

    @classmethod
    def create_from_params(cls, params, *args, **kwargs):
        return cls(params.num_epochs,
                   params.batch_size,
                   params.shard_name,
                   params.data_dir)

    def __init__(self,
                 batch_size,
                 num_epochs,
                 shard_name,
                 data_dir):
        # The single batch case needs obviously no shuffling.
        super(MnistSingleBatch, self).__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            shard_name=shard_name,
            shuffle=False,
            data_dir=data_dir)

    @property
    def num_examples(self):
        if self._shard_name == _interface.SHARD_NAME_TRAIN:
            return min(self._dataset.train.num_examples, self._batch_size)
        if self._shard_name == _interface.SHARD_NAME_VALIDATION:
            return min(self._dataset.validation.num_examples, self._batch_size)
        if self._shard_name == _interface.SHARD_NAME_TEST:
            return min(self._dataset.test.num_examples, self._batch_size)

    def _create_input_fn(self, dataset):
        # We only get the data once.
        images, labels = dataset.next_batch(self._batch_size)

        def input_fn():
            feature_dict = {'images': tf.constant(images,
                                                  dtype=tf.float32)}
            label_dict = {'labels': tf.constant(labels,
                                                dtype=tf.float32)}
            return feature_dict, label_dict
        return input_fn
