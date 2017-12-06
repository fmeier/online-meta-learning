
import numpy as np
import tensorflow as tf

from tf_utils import ops

from meta_learning.backend.tensorflow.memory import static as _static


class AdamStatic(_static.Static):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(params.memory_max_value,
                   params.memory_min_value,
                   params.memory_min_pred,
                   params.memory_max_pred,
                   params.memory_num_centers,
                   params.memory_lm_scale,
                   params.memory_predict_exp_smooth,
                   params.memory_learning_rate,
                   params.memory_beta_1,
                   params.memory_beta_2,
                   params.memory_epsilon,
                   params.learning_rate,
                   params.get('logdir', '/tmp/'),
                   params.get('is_debug', False))

    @classmethod
    def init_fn(cls,
                learning_rate,
                memory_clip_grad,
                memory_lm_scale,
                memory_num_centers,
                memory_learning_rate,
                memory_predict_exp_smooth=0.0,
                memory_beta_1=0.9,
                memory_beta_2=0.999,
                memory_epsilon=1e-8,
                memory_min_pred=1e-9,
                memory_max_pred=1.0,
                logdir=None,
                is_debug=False):
        def fn():
            return cls(memory_max_value=memory_clip_grad,
                       memory_min_value=-memory_clip_grad,
                       memory_min_pred=memory_min_pred,
                       memory_max_pred=memory_max_pred,
                       memory_num_centers=memory_num_centers,
                       memory_lm_scale=memory_lm_scale,
                       memory_predict_exp_smooth=memory_predict_exp_smooth,
                       memory_learning_rate=memory_learning_rate,
                       memory_beta_1=memory_beta_1,
                       memory_beta_2=memory_beta_2,
                       memory_epsilon=memory_epsilon,
                       learning_rate=learning_rate,
                       logdir=logdir,
                       is_debug=is_debug)
        return fn

    def __init__(self,
                 memory_max_value,
                 memory_min_value,
                 memory_min_pred,
                 memory_max_pred,
                 memory_num_centers,
                 memory_lm_scale,
                 memory_predict_exp_smooth,
                 memory_learning_rate,
                 memory_beta_1,
                 memory_beta_2,
                 memory_epsilon,
                 learning_rate,
                 logdir,
                 is_debug):
        super(AdamStatic, self).__init__(
            memory_max_value=memory_max_value,
            memory_min_value=memory_min_value,
            memory_min_pred=memory_min_pred,
            memory_max_pred=memory_max_pred,
            memory_num_centers=memory_num_centers,
            memory_lm_scale=memory_lm_scale,
            memory_predict_exp_smooth=memory_predict_exp_smooth,
            learning_rate=learning_rate,
            logdir=logdir,
            is_debug=is_debug)
        assert memory_learning_rate is not None, (
            'memory_learning_rate is not set.')
        assert memory_beta_1 is not None, (
            'memory_beta_1 is not set.')
        assert memory_beta_2 is not None, (
            'memory_beta_2 is not set.')
        assert memory_epsilon is not None, (
            'memory_epsilon is not set.')
        self._memory_learning_rate = np.array(
            memory_learning_rate, dtype=np.float32)
        self._memory_beta_1 = np.array(memory_beta_1, dtype=np.float32)
        self._memory_beta_2 = np.array(memory_beta_2, dtype=np.float32)
        self._memory_epsilon = np.array(memory_epsilon, dtype=np.float32)

    def create_variables(self, variable, optimizer):
        super(AdamStatic, self).create_variables(variable, optimizer)

        zeros = np.zeros(self._num_centers, dtype=np.float32)

        self._add_to_slots(optimizer,
                           variable,
                           'memory_grad_prev_mean_tf',
                           zeros)

        self._add_to_slots(optimizer,
                           variable,
                           'memory_grad_prev_var_mean_tf',
                           zeros)

        self._add_to_slots(optimizer,
                           variable,
                           'memory_beta_1_power',
                           self._memory_beta_1)

        self._add_to_slots(optimizer,
                           variable,
                           'memory_beta_2_power',
                           self._memory_beta_2)

        self._memory_learning_rate = tf.constant(self._memory_learning_rate,
                                                 name='learning_rate')
        # The constants have to be created after we use the members for slots.
        self._memory_beta_1 = tf.constant(self._memory_beta_1, name='beta_1')
        self._memory_beta_2 = tf.constant(self._memory_beta_2, name='beta_2')
        self._memory_epsilon = tf.constant(
            self._memory_epsilon, name='epsilon')

    def _get_grad_apply(self, grad):
        grad_cur_mean_tf = ops.exp_smooth(
            self._memory_beta_1,
            self.memory_grad_prev_mean_tf,
            grad)
        grad_cur_var_mean_tf = ops.exp_smooth(
            self._memory_beta_2,
            self.memory_grad_prev_var_mean_tf,
            tf.square(grad))

        learning_rate = self._memory_learning_rate * tf.divide(
            tf.sqrt(1 - self.memory_beta_2_power),
            1 - self.memory_beta_1_power)

        grad_change = tf.divide(grad_cur_mean_tf,
                                tf.add(tf.sqrt(grad_cur_var_mean_tf),
                                       self._memory_epsilon))

        grad_apply = tf.multiply(learning_rate, grad_change)

        with tf.control_dependencies([grad_apply]):
            assign_ops = []
            assign_ops.append(
                self.memory_grad_prev_mean_tf.assign(grad_cur_mean_tf))
            assign_ops.append(
                self.memory_grad_prev_var_mean_tf.assign(grad_cur_var_mean_tf))
            assign_ops.append(self.memory_beta_1_power.assign(
                tf.multiply(self.memory_beta_1_power, self._memory_beta_1)))
            assign_ops.append(self.memory_beta_2_power.assign(
                tf.multiply(self.memory_beta_2_power, self._memory_beta_2)))
            return grad_apply, assign_ops
