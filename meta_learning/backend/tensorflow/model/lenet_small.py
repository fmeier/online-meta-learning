"""A very simple model.
"""

import tensorflow as tf

from tf_utils import variables as _variables
from tf_utils import summaries as _summaries
from tf_utils.model import interface as _interface


DEFAULT_FILTER_SIZE = [8, 128]


class LenetSmall(_interface.Interface):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(
            model_dir=params.model_dir,
            filter_size=params.get('filter_size', None),
            use_exclude_var_names=params.get(
                'use_exclude_var_names', False))

    def __init__(self,
                 model_dir,
                 filter_size,
                 use_exclude_var_names=False):
        super(ModelLenetSmall, self).__init__(
            model_dir=model_dir,
            use_exclude_var_names=use_exclude_var_names)
        if filter_size is None:
            filter_size = DEFAULT_FILTER_SIZE
        else:
            assert len(filter_size) == len(DEFAULT_FILTER_SIZE)
        self._filter_size = filter_size

    def create_model_fn(self,
                        optimizer_fn,
                        saver_fn=None):
        """Our model function has to return an estimator_spec.
        """
        def model_fn(mode, features, labels):
            estimator_spec = self._get_estimator_spec_dict(mode)

            images = features['images']
            labels = labels['labels']

            net = tf.layers.conv2d(images,
                                   filters=self._filter_size[0],
                                   kernel_size=[5, 5],
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv_1')
            net = tf.layers.max_pooling2d(net,
                                          pool_size=[2, 2],
                                          strides=[2, 2],
                                          padding='same',
                                          name='max_1')
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net,
                                  self._filter_size[1],
                                  activation=tf.nn.relu,
                                  name='fc_2')

            saver = None
            if saver_fn is not None:
                saver = saver_fn()

            if mode == tf.estimator.ModeKeys.TRAIN:
                # dropout scales automatically such that it is not required
                # at test time.
                net = tf.layers.dropout(net, rate=0.5)

            logits = tf.layers.dense(net,
                                     labels.get_shape().as_list()[1],
                                     activation=None,
                                     name='fc_3')

            self._init_fn_exclude_var_names.extend(
                _variables.get_variable_names_in_scope())

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = optimizer_fn()
                tf.losses.softmax_cross_entropy(labels, logits)

                estimator_spec['loss'] = tf.losses.get_total_loss()
                tf.summary.scalar('loss', estimator_spec['loss'])

                estimator_spec['train_op'] = optimizer.create_train_op(
                    estimator_spec['loss'])

                summary = _summaries.SaveTensorDict(
                    self._model_dir,
                    'lenet_small',
                    only_if_cpu=True)
                summary.create_save_summary(
                    {'loss': estimator_spec['loss']})

                # Notice the hooks have to be added after all the other
                # ops otherwise the saver cannot load everything.
                logging_dict = {
                    'logging_loss': estimator_spec['loss'],
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
                estimator_spec['predictions'] = {'logits':
                                                 logits}
            return tf.estimator.EstimatorSpec(**estimator_spec)
        return model_fn
