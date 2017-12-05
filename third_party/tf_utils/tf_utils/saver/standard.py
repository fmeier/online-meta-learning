

import abc
import tensorflow as tf

from tf_utils import checkpoint as _checkpoint
from tf_utils.saver import interface as _interface


class Standard(_interface.Interface):
    """Interface for our gradient object
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(params.init_checkpoint_dir,
                   params.checkpoint_save_steps,
                   params.checkpoint_max_to_keep,
                   params.summary_save_steps,
                   params.logging_steps)

    def __init__(self,
                 init_checkpoint_dir,
                 checkpoint_save_steps,
                 checkpoint_max_to_keep,
                 summary_save_steps,
                 logging_steps):
        self._init_checkpoint_dir = init_checkpoint_dir
        self._checkpoint_save_steps = checkpoint_save_steps
        self._checkpoint_max_to_keep = checkpoint_max_to_keep
        self._summary_save_steps = summary_save_steps
        self._logging_steps = logging_steps

    @classmethod
    def init_fn(cls,
                init_checkpoint_dir=None,
                checkpoint_save_steps=20,
                checkpoint_max_to_keep=4,
                summary_save_steps=1,
                logging_steps=1):
        def fn():
            return cls(init_checkpoint_dir,
                       checkpoint_save_steps,
                       checkpoint_max_to_keep,
                       summary_save_steps,
                       logging_steps)
        return fn

    def create_training_hooks(self,
                              model_dir,
                              logging_dict):
        training_hooks = []

        training_hooks.append(tf.train.CheckpointSaverHook(
            checkpoint_dir=model_dir,
            save_steps=self._checkpoint_save_steps,
            saver=tf.train.Saver(max_to_keep=self._checkpoint_max_to_keep)
        ))

        training_hooks.append(tf.train.LoggingTensorHook(
            logging_dict,
            every_n_iter=self._logging_steps))

        summary_op = tf.summary.merge_all()
        if summary_op is not None:
            training_hooks.append(tf.train.SummarySaverHook(
                save_steps=self._summary_save_steps,
                output_dir=model_dir,
                summary_op=summary_op
            ))
        return training_hooks

    def create_scaffold(self, exclude_var_names):
        with tf.variable_scope('scaffold'):
            init_fn = _checkpoint.create_init_fn(self._init_checkpoint_dir,
                                                 exclude_var_names)
            if init_fn is not None:
                def fn(_, sess):
                    init_fn(sess)
                return tf.train.Scaffold(
                    init_fn=fn)
            return None
