
import abc

import numpy as np
import tensorflow as tf

from tf_utils import debug as _debug
from tf_utils import ops as _tf_utils_ops
from tf_utils import summaries as _summaries

from meta_learning.backend.tensorflow.memory import interface as _interface
from meta_learning.backend.tensorflow.memory import ops as _ops


class Static(_interface.Interface):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        # This is an abstract class and should not be instantiated.
        raise NotImplementedError()

    def __init__(self,
                 memory_max_value,
                 memory_min_value,
                 memory_min_pred,
                 memory_max_pred,
                 memory_num_centers,
                 memory_activation_threshold,
                 memory_predict_exp_smooth,
                 learning_rate,
                 logdir,
                 is_debug):
        super(Static, self).__init__()
        self._max_value = np.array(memory_max_value, dtype=np.float32)
        self._min_value = np.array(memory_min_value, dtype=np.float32)
        self._min_pred = np.array(memory_min_pred, dtype=np.float32)
        self._max_pred = np.array(memory_max_pred, dtype=np.float32)
        self._num_centers = np.array(memory_num_centers, dtype=np.int32)
        self._learning_rate = np.array(learning_rate, dtype=np.float32)
        self._predict_exp_smooth = np.array(
            memory_predict_exp_smooth, dtype=np.float32)
        self._logdir = logdir
        self._is_debug = is_debug
        self._centers = np.linspace(self._min_value,
                                    self._max_value,
                                    self._num_centers).astype(np.float32)
        self._activation_threshold = memory_activation_threshold
        self._name = ''

    @abc.abstractmethod
    def _get_grad_apply(self, grad_change):
        raise NotImplementedError()

    def create_variables(self, variable, optimizer):
        self._name = variable.name
        self._dtype = variable.dtype
        self._variable = variable
        variable_size = variable.get_shape().as_list()
        if variable_size:
            zeros_variable = np.zeros(
                np.prod(variable_size), dtype=np.float32)
        else:
            zeros_variable = np.zeros(
                1, dtype=np.float32)

        # Setting sigma according to distance half squared
        self._threshold_activation_tf = tf.constant(
            self._activation_threshold, dtype=self._dtype)
        self._sigma_tf = tf.constant(
            ((self._centers[0] - self._centers[1]) * 2.0)**2,
            dtype=self._dtype)
        self._center_tf = tf.constant(self._centers, dtype=self._dtype)
        self._ones = tf.ones_like(self._center_tf, dtype=self._dtype)
        init_value = np.ones(
            self._num_centers, dtype=np.float32) * self._learning_rate
        self._add_to_slots(optimizer,
                           variable,
                           'value_tf',
                           init_value,
                           exclude_var_names=False)
        self._add_to_slots(optimizer,
                           variable,
                           'inference_exp_smooth_tf',
                           zeros_variable)

    def compute_activation(self, grad_feature):
        grad_feature = tf.clip_by_value(grad_feature, self._min_value,
                                        self._max_value)

        activation = _ops.multi_activation_rbf(
            grad_feature, self._center_tf, self._sigma_tf)
        return activation

    def update_memory(self, grad_cur, grad_prev, activation):
        grad_cur = tf.clip_by_value(grad_cur,
                                    self._min_value,
                                    self._max_value)
        grad_prev = tf.clip_by_value(grad_prev,
                                     self._min_value,
                                     self._max_value)
        grad_cur = tf.reshape(grad_cur, [-1])
        grad_prev = tf.reshape(grad_prev, [-1])

        #####################################################################
        # we compute the gradient based update change
        #####################################################################
        with tf.control_dependencies(
                [grad_cur, grad_prev, activation]):

            grad_change = _ops.gradient_sign_by_prod(
                grad_cur, grad_prev, activation)

            with tf.control_dependencies([grad_change]):
                apply_add, assign_ops = self._get_grad_apply(grad_change)

                value_updated = self.value_tf + apply_add

                value_updated = tf.clip_by_value(value_updated,
                                                 self._min_pred,
                                                 self._max_pred)
                if self._is_debug:
                    saver = _summaries.SaveTensorDict(
                        self._logdir,
                        self._name + '_memory',
                        only_if_cpu=True)
                    debug_tensors = {}
                    debug_tensors['param_val'] = self._variable
                    debug_tensors['grad_prev'] = grad_prev
                    debug_tensors['grad_cur'] = grad_cur
                    debug_tensors['grad_change'] = grad_change
                    debug_tensors['activation'] = activation
                    debug_tensors['values_prev'] = self.value_tf
                    debug_tensors['values'] = value_updated
                    debug_tensors['centers'] = self._center_tf
                    debug_tensors['sigmasq'] = self._sigma_tf
                    summary_op = saver.create_save_summary(debug_tensors)

                    with tf.control_dependencies([summary_op]):
                        assign_ops.append(self.value_tf.assign(value_updated))
                    return assign_ops
                assign_ops.append(self.value_tf.assign(value_updated))
                return assign_ops

    def _inference(self, activation):
        inference_ops = {}
        prediction = _ops.multi_compute_eta(activation,
                                            self.value_tf,
                                            is_debug=self._is_debug)

        # beta*new + (1-beta)*old
        prediction_mean_tf = _tf_utils_ops.exp_smooth(
            self._predict_exp_smooth,
            self.inference_exp_smooth_tf, prediction)
        with tf.control_dependencies(
                [self.inference_exp_smooth_tf.assign(prediction_mean_tf)]):
            prediction = tf.identity(prediction_mean_tf)
        inference_ops['learning_rate'] = prediction
        return inference_ops
