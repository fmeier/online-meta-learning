
import numpy as np
import tensorflow as tf

from tf_utils import device as _device
from tf_utils.optimizer import interface as _interface


class Standard(_interface.Interface):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        # This is an abstract class and should not be instantiated.
        raise NotImplementedError()

    def __init__(self,
                 learning_rate,
                 use_exponential_decay,
                 decay_steps,
                 decay_rate,
                 use_exclude_var_names=False):
        super(Standard, self).__init__(
            use_exclude_var_names=use_exclude_var_names)
        self._learning_rate = np.array(learning_rate, dtype=np.float32)
        self._use_exponential_decay = use_exponential_decay
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate

    def _get_learning_rate(self):
        if self._use_exponential_decay:
            if self._decay_steps is None:
                raise Exception(
                    'decay steps have to be defined.')
            if self._decay_rate is None:
                raise Exception(
                    'decay rate have to be defined.')
            global_step = tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                self._learning_rate, global_step,
                self._decay_steps, self._decay_rate, staircase=True)
        else:
            learning_rate = tf.constant(self._learning_rate)
        with _device.device_cpu():
            tf.summary.scalar('learning_rate', learning_rate)
        return learning_rate
