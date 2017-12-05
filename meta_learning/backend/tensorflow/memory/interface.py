
import abc

import tensorflow as tf

from tf_utils import factory as _factory


class Interface(object):
    """Interface for our memory object.
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return _factory.create_from_params(cls,
                                           key_class_type='memory',
                                           params=params,
                                           **kwargs)

    def __init__(self):
        self._slots = []
        self._init_fn_exclude_var_names = []

    def update_slots(self, optimizer, variable):
        for slot_name in self._slots:
            setattr(self, slot_name, optimizer.get_slot(variable, slot_name))

    def train(self, gradient_dict):
        with tf.variable_scope('train', values=gradient_dict.values()):
            activation = self.compute_activation(gradient_dict['grad_prev_tf'])
            with tf.control_dependencies([activation]):
                return self.update_memory(
                    gradient_dict['grad_cur_tf'],
                    gradient_dict['grad_prev_tf'],
                    activation)

    def inference(self, gradient_dict):
        with tf.variable_scope('inference', values=gradient_dict.values()):
            activation = self.compute_activation(gradient_dict['grad_cur_tf'])
            return self._inference(activation)

    @abc.abstractmethod
    def create_variables(self, variable, optimizer):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_activation(self, grad_feature):
        raise NotImplementedError

    @abc.abstractmethod
    def update_memory(self, gradient_dict, grad_data, activation):
        raise NotImplementedError

    @abc.abstractmethod
    def _inference(self, activation):
        raise NotImplementedError

    def init_fn_exclude_var_names(self):
        return self._init_fn_exclude_var_names

    def _add_to_slots(self,
                      optimizer,
                      variable,
                      name,
                      init_value,
                      init_dtype=None,
                      exclude_var_names=True):
        self._slots.append(name)
        if init_dtype is None:
            init_dtype = init_value.dtype
        with tf.variable_scope(name):
            var = optimizer._get_or_make_slot_with_initializer(
                variable,
                tf.constant_initializer(init_value),
                tf.TensorShape(init_value.shape),
                init_dtype,
                name,
                optimizer._name)
            if exclude_var_names:
                self._init_fn_exclude_var_names.append(var.op.name)
