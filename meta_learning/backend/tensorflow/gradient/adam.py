import numpy as np
import tensorflow as tf

from tf_utils import ops

from meta_learning.backend.tensorflow.gradient import interface as _interface


class Adam(_interface.Interface):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(params.learning_rate,
                   params.beta_1,
                   params.beta_2,
                   params.epsilon,
                   params.get('clip_by_value', None),
                   params.get('is_debug', False),
                   params.get('logdir', '/tmp'))

    def __init__(self,
                 learning_rate,
                 beta_1,
                 beta_2,
                 epsilon,
                 clip_by_value,
                 is_debug,
                 logdir):
        super(Adam, self).__init__(clip_by_value)
        self._learning_rate = np.array(learning_rate, dtype=np.float32)
        self._beta_1 = np.array(beta_1, dtype=np.float32)
        self._beta_2 = np.array(beta_2, dtype=np.float32)
        self._epsilon = np.array(epsilon, dtype=np.float32)
        self._is_debug = is_debug

    def create_variables(self, variable, optimizer):
        variable_shape = tf.reshape(variable, [-1]).get_shape().as_list()
        # variable_shape = variable.get_shape().as_list()
        zeros = np.zeros(variable_shape, dtype=np.float32)

        self._add_to_slots(optimizer,
                           variable,
                           'grad_prev_raw_tf',
                           zeros)

        self._add_to_slots(optimizer,
                           variable,
                           'grad_prev_mean_tf',
                           zeros)

        self._add_to_slots(optimizer,
                           variable,
                           'grad_prev_var_raw_tf',
                           zeros)

        self._add_to_slots(optimizer,
                           variable,
                           'grad_prev_var_mean_tf',
                           zeros)

        self._add_to_slots(optimizer,
                           variable,
                           'beta_1_power',
                           self._beta_1)

        self._add_to_slots(optimizer,
                           variable,
                           'beta_2_power',
                           self._beta_2)

        self._learning_rate = tf.constant(self._learning_rate,
                                          name='learning_rate')
        # The constants have to be created after we use the members for slots.
        self._beta_1 = tf.constant(self._beta_1, name='beta_1')
        self._beta_2 = tf.constant(self._beta_2, name='beta_2')
        self._epsilon = tf.constant(self._epsilon, name='epsilon')

    def update_statistics(self, grad):
        gradient_ops = {}
        with tf.variable_scope('update_statistics', values=[grad]):
            gradient_ops['grad_prev_raw_tf'] = self.grad_prev_raw_tf
            gradient_ops['grad_prev_mean_tf'] = self.grad_prev_mean_tf
            gradient_ops['grad_prev_var_raw_tf'] = self.grad_prev_var_raw_tf
            gradient_ops['grad_prev_var_mean_tf'] = self.grad_prev_var_mean_tf

            gradient_ops['grad_cur_raw_tf'] = grad
            gradient_ops['grad_cur_mean_tf'] = ops.exp_smooth(
                self._beta_1,
                self.grad_prev_mean_tf,
                gradient_ops['grad_cur_raw_tf'])
            gradient_ops['grad_cur_var_raw_tf'] = tf.square(grad)
            gradient_ops['grad_cur_var_mean_tf'] = ops.exp_smooth(
                self._beta_2,
                self.grad_prev_var_mean_tf,
                gradient_ops['grad_cur_var_raw_tf'])

            # This is the interface for the external usage.
            gradient_ops['grad_prev_tf'] = gradient_ops['grad_prev_mean_tf']
            gradient_ops['grad_cur_tf'] = gradient_ops['grad_cur_mean_tf']
            return gradient_ops

    def compute_apply(self, gradient_ops, memory_inference_ops):
        with tf.variable_scope('compute_apply',
                               values=(gradient_ops.values() +
                                       memory_inference_ops.values())):

            grad_apply = tf.divide(
                gradient_ops['grad_cur_tf'],
                tf.add(
                    tf.sqrt(gradient_ops['grad_cur_var_mean_tf']),
                    self._epsilon))
            grad_apply = self._clip_grad(grad_apply)

            if 'learning_rate' in memory_inference_ops:
                learning_rate = memory_inference_ops['learning_rate']
            else:
                learning_rate = self._learning_rate * tf.divide(
                    tf.sqrt(1 - self.beta_2_power),
                    1 - self.beta_1_power)

            if learning_rate.get_shape().ndims != 0:
                learning_rate = tf.reshape(learning_rate,
                                           grad_apply.get_shape())

            with tf.control_dependencies([learning_rate]):
                grad_apply = tf.multiply(grad_apply, learning_rate)
            with tf.control_dependencies([grad_apply]):
                assign_ops = self._get_assign_ops(
                    grad_apply, gradient_ops)
            return grad_apply, assign_ops

    def _get_assign_ops(self, grad_apply, gradient_ops):
        assign_ops = []
        update_op_names = ['grad_cur_raw_tf',
                           'grad_cur_mean_tf',
                           'grad_cur_var_raw_tf',
                           'grad_cur_var_mean_tf']
        for update_op_name in update_op_names:
            update_op_name_prev = update_op_name.replace('cur', 'prev')
            assign_ops.append(getattr(self, update_op_name_prev).assign(
                gradient_ops[update_op_name]))
        assign_ops.append(self.beta_1_power.assign(
            tf.multiply(self.beta_1_power, self._beta_1)))
        assign_ops.append(self.beta_2_power.assign(
            tf.multiply(self.beta_2_power, self._beta_2)))
        return assign_ops
