
import numpy as np
import tensorflow as tf


from tf_utils import summaries as _summaries
from tf_utils.model import interface as _interface


class Rosenbrock(_interface.Interface):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(params.model_dir,
                   params.joined,
                   params.trainable,
                   use_exclude_var_names=params.get(
                       'use_exclude_var_names', False))

    def __init__(self,
                 model_dir,
                 joined,
                 trainable,
                 use_exclude_var_names=False):
        super(Rosenbrock, self).__init__(
            model_dir=model_dir,
            use_exclude_var_names=use_exclude_var_names)
        self._joined = joined
        self._trainable = trainable

    def create_model_fn(self,
                        optimizer_fn,
                        saver_fn=None):
        """Our model function has to return an estimator_spec.
        """

        def model_fn(mode, features, labels):

            estimator_spec = self._get_estimator_spec_dict(mode)

            with tf.variable_scope(self._get_scope_name('variables')):
                initial_params = np.array([1.5, -0.5], dtype=np.float32)
                # We have one gradient
                if self._joined:
                    x = tf.get_variable(
                        'x',
                        initializer=initial_params,
                        trainable=True)
                    x1 = x[0]
                    x2 = x[1]
                    self._init_fn_exclude_var_names.append(x.op.name)
                else:
                    x1 = tf.get_variable(
                        'x1',
                        initializer=initial_params[0],
                        trainable=True)  # self._trainable.x1)
                    self._init_fn_exclude_var_names.append(x1.op.name)
                    x2 = tf.get_variable(
                        'x2',
                        initializer=initial_params[1],
                        trainable=True)  # self._trainable.x2)
                    self._init_fn_exclude_var_names.append(x2.op.name)

            with tf.variable_scope(self._get_scope_name('rosenbrock')):
                # (p.a - x[1])^2 + p.b * (x[2] - x[1]^2)^2;
                rosenbrock = (
                    tf.square(1.0 - x1) +
                    100.0 * tf.square(x2 - tf.square(x1)))
                rosenbrock = tf.Print(rosenbrock, [rosenbrock], 'rosenbrock')

            saver = None
            if saver_fn is not None:
                saver = saver_fn()

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = optimizer_fn()

                # We simply want to optimize the rosenbrock function
                estimator_spec['loss'] = rosenbrock
                tf.summary.scalar('loss', estimator_spec['loss'])

                estimator_spec['train_op'] = optimizer.create_train_op(
                    estimator_spec['loss'])

                summary = _summaries.SaveTensorDict(
                    self._model_dir,
                    'rosenbrock',
                    only_if_cpu=True)
                summary.create_save_summary(
                    {'loss': estimator_spec['loss'],
                     'x1': x1,
                     'x2': x2})

                # Notice the hooks have to be added after all the other
                # ops otherwise the saver cannot load everything.
                logging_dict = {
                    'logging_loss': rosenbrock,
                    'step': tf.contrib.framework.get_or_create_global_step()}
                if saver is not None:
                    estimator_spec['training_hooks'].extend(
                        saver.create_training_hooks(
                            self._model_dir, logging_dict))
                    estimator_spec['scaffold'] = self.create_scaffold(
                        saver, optimizer)

            if mode == tf.estimator.ModeKeys.EVAL:
                pass
            if mode == tf.estimator.ModeKeys.PREDICT:
                estimator_spec['predictions'] = {'rosenbrock':
                                                 rosenbrock}
            return tf.estimator.EstimatorSpec(**estimator_spec)
        return model_fn
