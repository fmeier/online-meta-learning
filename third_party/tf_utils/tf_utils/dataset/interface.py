
import abc

from tf_utils import factory as _factory

SHARD_NAME_TRAIN = 'train'
SHARD_NAME_VALIDATION = 'validation'
SHARD_NAME_TEST = 'test'


class Interface(object):

    __metaclass__ = abc.ABCMeta

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return _factory.create_from_params(cls,
                                           key_class_type='dataset',
                                           params=params,
                                           **kwargs)

    def __init__(self,
                 batch_size,
                 num_epochs,
                 shard_name):
        super(Interface, self).__init__()
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._shard_name = shard_name
        self._exclude_var_names = []
        if self._shard_name not in self.shard_names():
            raise Exception('shard_name {} not known, we only know {}.'.format(
                self._shard_name, self.shard_names()))

    def shard_names(self):
        return [SHARD_NAME_TRAIN, SHARD_NAME_VALIDATION, SHARD_NAME_TEST]

    @abc.abstractproperty
    def num_examples(self):
        raise NotImplementedError()

    @property
    def steps(self):
        return self.num_examples / self._batch_size * self._num_epochs

    @abc.abstractmethod
    def create_input_fn(self):
        raise NotImplementedError()
