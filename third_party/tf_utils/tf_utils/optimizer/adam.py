
import numpy as np
import tensorflow as tf

from tf_utils import variables as _variables
from tf_utils.optimizer import standard as _standard


class Adam(_standard.Standard):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(
            learning_rate=params.learning_rate,
            use_exponential_decay=params.get('use_exponential_decay', False),
            decay_steps=params.get('decay_steps', None),
            decay_rate=params.get('decay_rate', None),
            beta_1=params.beta_1,
            beta_2=params.beta_2,
            epsilon=params.epsilon,
            use_exclude_var_names=params.get(
                'use_exclude_var_names', False))

    def __init__(self,
                 learning_rate,
                 use_exponential_decay,
                 decay_steps,
                 decay_rate,
                 beta_1,
                 beta_2,
                 epsilon,
                 use_exclude_var_names=False):
        super(Adam, self).__init__(
            learning_rate=learning_rate,
            use_exponential_decay=use_exponential_decay,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            use_exclude_var_names=use_exclude_var_names)
        self._beta_1 = np.array(beta_1, dtype=np.float32)
        self._beta_2 = np.array(beta_2, dtype=np.float32)
        self._epsilon = np.array(epsilon, dtype=np.float32)

    def get_optimizer_op(self, global_step):
        return tf.train.AdamOptimizer(
            learning_rate=self._get_learning_rate(),
            beta1=self._beta_1,
            beta2=self._beta_2,
            epsilon=self._epsilon)

    def _update_init_fn_exclude_var_names(self):
        for var_name in _variables.get_variable_names_in_scope():
            if 'Adam' in var_name:
                self._init_fn_exclude_var_names.append(var_name)
        print(self._init_fn_exclude_var_names)
