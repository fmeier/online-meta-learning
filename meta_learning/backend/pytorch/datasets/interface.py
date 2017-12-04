import abc

SHARD_NAME_TRAIN = 'train'
SHARD_NAME_VALIDATION = 'validation'
SHARD_NAME_TEST = 'test'


class Interface(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 batch_size,
                 shard_name):
        super(Interface, self).__init__()
        self._batch_size = batch_size
        self._shard_name = shard_name
        if self._shard_name not in self.shard_names():
            raise Exception('shard_name {} not known, we only know {}.'.format(
                self._shard_name, self.shard_names()))

    def shard_names(self):
        return [SHARD_NAME_TRAIN, SHARD_NAME_VALIDATION, SHARD_NAME_TEST]

    @abc.abstractproperty
    def num_examples(self):
        raise NotImplementedError()

    # @property
    # def steps(self):
    #     return self.num_examples / self._batch_size * self._num_epochs

    @abc.abstractmethod
    def get_next_batch(self):
        raise NotImplementedError()
