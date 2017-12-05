import abc

import tensorflow as tf

from tf_utils import factory as _factory


class Interface(object):

    __metaclass__ = abc.ABCMeta

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return _factory.create_from_params(cls,
                                           key_class_type='optimizer',
                                           params=params,
                                           **kwargs)

    def __init__(self,
                 use_exclude_var_names):
        super(Interface, self).__init__()
        self._init_fn_exclude_var_names = []
        self._use_exclude_var_names = use_exclude_var_names

    @abc.abstractmethod
    def get_optimizer_op(self, global_step):
        raise NotImplementedError()

    def init_fn_exclude_var_names(self):
        """A list of names which are excluded when loading a model.

        The default None means everything is loaded, nothing is excluded.
        """
        if self._use_exclude_var_names:
            return self._init_fn_exclude_var_names
        return []

    def _get_scope_name(self, name):
        return self.__class__.__name__ + '/' + name

    def _update_init_fn_exclude_var_names(self):
        pass

    def create_train_op(self, loss):
        with tf.variable_scope(self._get_scope_name('train_op')):
            train_op = tf.contrib.training.create_train_op(
                loss,
                self.get_optimizer_op(
                    tf.train.get_or_create_global_step()))
            self._update_init_fn_exclude_var_names()
            return train_op
