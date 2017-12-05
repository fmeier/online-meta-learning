
import abc
from tf_utils import factory as _factory


class Interface(object):
    """Interface for our gradient object
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return _factory.create_from_params(cls,
                                           key_class_type='saver',
                                           params=params,
                                           **kwargs)

    def __init__(self):
        pass

    @abc.abstractmethod
    def create_training_hooks(self, model_dir):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_scaffold(self, exclude_var_names):
        raise NotImplementedError()
