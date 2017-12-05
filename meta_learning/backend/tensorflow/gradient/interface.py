import abc
import numpy as np
import tensorflow as tf
from tf_utils import factory as _factory


class Interface(object):
    """Interface for our gradient object
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return _factory.create_from_params(cls,
                                           key_class_type='gradient',
                                           params=params,
                                           **kwargs)

    def __init__(self, clip_by_value):
        self._slots = []
        self._init_fn_exclude_var_names = []
        self._clip_by_value = None
        if clip_by_value is not None and clip_by_value != 'NOT_SET':
            self._clip_by_value = np.array(clip_by_value, dtype=np.float32)

    def _clip_grad(self, grad_apply):
        if self._clip_by_value is not None:
            return tf.clip_by_value(grad_apply,
                                    -self._clip_by_value,
                                    self._clip_by_value)
        return grad_apply

    def update_slots(self, optimizer, variable):
        # By default we do not need slots
        for slot_name in self._slots:
            setattr(self, slot_name, optimizer.get_slot(variable, slot_name))

    @abc.abstractmethod
    def create_variables(self, variable, optimizer):
        raise NotImplementedError

    @abc.abstractmethod
    def update_statistics(self, gradient):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_apply(self, gradient_ops, memory_inference_ops):
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
