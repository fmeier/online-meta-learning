
import numpy as np
import tensorflow as tf

from tf_utils import ops

from meta_learning.backend.tensorflow.memory import static as _static


class MomentumStatic(_static.Static):

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
                   params.memory_momentum,
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
                memory_momentum=0.9,
                memory_predict_exp_smooth=0.0,
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
                       memory_momentum=memory_momentum,
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
                 memory_momentum,
                 learning_rate,
                 logdir,
                 is_debug):
        super(MomentumStatic, self).__init__(
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
        self._learning_rate = np.array(learning_rate, dtype=np.float32)
        self._memory_momentum = np.array(memory_momentum, dtype=np.float32)
        self._memory_learning_rate = np.array(
            memory_learning_rate, dtype=np.float32)

    def create_variables(self, variable, optimizer):
        super(MomentumStatic, self).create_variables(variable, optimizer)

        zeros = np.zeros(self._num_centers, dtype=np.float32)

        self._add_to_slots(optimizer,
                           variable,
                           'memory_grad_mean_tf',
                           zeros)

        # The constants have to be created after we use the members for slots.
        self._memory_momentum = tf.constant(self._memory_momentum, name='mom')

    def _get_grad_apply(self, grad):
        grad_change = ops.exp_smooth(
            self._memory_momentum,
            self.memory_grad_mean_tf,
            grad)

        grad_apply = tf.multiply(self._memory_learning_rate, grad_change)

        with tf.control_dependencies([grad_apply]):
            assign_ops = []
            assign_ops.append(
                self.memory_grad_mean_tf.assign(grad_change))
            return grad_apply, assign_ops
