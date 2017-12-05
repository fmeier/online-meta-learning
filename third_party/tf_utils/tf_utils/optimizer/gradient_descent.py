

import tensorflow as tf

from tf_utils.optimizer import standard as _standard


class GradientDescent(_standard.Standard):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(
            learning_rate=params.learning_rate,
            use_exponential_decay=params.get('use_exponential_decay', False),
            decay_steps=params.get('decay_steps', None),
            decay_rate=params.get('decay_rate', None),
            use_exclude_var_names=params.get(
                'use_exclude_var_names', False))

    def __init__(self,
                 learning_rate,
                 use_exponential_decay,
                 decay_steps,
                 decay_rate,
                 use_exclude_var_names=False):
        super(GradientDescent, self).__init__(
            learning_rate=learning_rate,
            use_exponential_decay=use_exponential_decay,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            use_exclude_var_names=use_exclude_var_names)

    def get_optimizer_op(self, global_step):
        return tf.train.GradientDescentOptimizer(
            learning_rate=self._get_learning_rate())
