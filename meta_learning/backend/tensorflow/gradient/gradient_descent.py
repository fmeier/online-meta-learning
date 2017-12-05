import numpy as np
import tensorflow as tf

from tf_utils import summaries

from meta_learning.backend.tensorflow.gradient import interface as _interface


class GradientDescent(_interface.Interface):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(params.learning_rate,
                   params.get('clip_by_value', None),
                   params.get('is_debug', False),
                   params.get('logdir', '/tmp'))

    def __init__(self,
                 learning_rate,
                 clip_by_value,
                 is_debug,
                 logdir):
        super(GradientDescent, self).__init__(clip_by_value)
        self._learning_rate = np.array(learning_rate, dtype=np.float32)
        self._is_debug = is_debug
        self._logdir = logdir

    @classmethod
    def init_fn(cls,
                learning_rate,
                clip_by_value,
                is_debug=False,
                logdir=None):
        def fn():
            return cls(learning_rate,
                       clip_by_value,
                       is_debug=False,
                       logdir=None)
        return fn

    def create_variables(self, variable, optimizer):
        variable_shape = tf.reshape(variable, [-1]).get_shape().as_list()
        # variable_shape = variable.get_shape().as_list()
        zeros = np.zeros(variable_shape, dtype=np.float32)
        self._learning_rate = tf.constant(self._learning_rate,
                                          name='learning_rate')
        self._name = variable.name

        self._add_to_slots(optimizer,
                           variable,
                           'grad_prev_raw_tf',
                           zeros)

    def update_statistics(self, grad):
        gradient_ops = {}
        with tf.variable_scope('update_statistics', values=[grad]):
            gradient_ops['grad_prev_tf'] = self.grad_prev_raw_tf
            gradient_ops['grad_cur_tf'] = grad
            return gradient_ops

    def compute_apply(self, gradient_ops, memory_inference_ops):
        with tf.variable_scope('compute_apply',
                               values=(memory_inference_ops.values())):

            grad_apply = gradient_ops['grad_cur_tf']
            grad_apply = self._clip_grad(grad_apply)

            learning_rate = self._learning_rate
            if 'learning_rate' in memory_inference_ops:
                learning_rate = memory_inference_ops['learning_rate']
            if learning_rate.get_shape().ndims != 0:
                learning_rate = tf.reshape(learning_rate,
                                           grad_apply.get_shape())

            if self._is_debug:
                saver = summaries.SaveTensorDict(
                    self._logdir,
                    self._name + 'gradient_info',
                    only_if_cpu=True)
                input_tensors = {}
                input_tensors['grad_cur'] = gradient_ops['grad_cur_tf']
                input_tensors['learning_rate'] = learning_rate
                saver.create_save_summary(input_tensors)

            grad_apply = tf.multiply(grad_apply, learning_rate)

            with tf.control_dependencies([grad_apply]):
                assign_ops = self._get_assign_ops(
                    grad_apply, gradient_ops)

            return grad_apply, assign_ops

    def _get_assign_ops(self, grad_apply, gradient_ops):
        grad_assign_ops = []
        grad_assign_ops.append(self.grad_prev_raw_tf.assign(
            gradient_ops['grad_cur_tf']))
        return grad_assign_ops
