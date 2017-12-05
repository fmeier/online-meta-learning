
import abc


import tensorflow as tf

from tf_utils import factory as _factory


class Interface(object):
    """Interface for our models.
    """

    __metaclass__ = abc.ABCMeta

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return _factory.create_from_params(cls,
                                           key_class_type='model',
                                           params=params,
                                           **kwargs)

    def __init__(self,
                 model_dir,
                 use_exclude_var_names):
        super(Interface, self).__init__()
        self._model_dir = model_dir
        self._init_fn_exclude_var_names = []
        self._use_exclude_var_names = use_exclude_var_names

    def create_estimator(self, model_fn, config=None, params=None):
        return tf.estimator.Estimator(model_fn=model_fn,
                                      model_dir=self._model_dir,
                                      config=config,
                                      params=params)

    @abc.abstractmethod
    def create_model_fn(self, optimizer_fn, saver_fn=None):
        """Our model function has to return an estimator_spec.
        """
        raise NotImplementedError()

    def _get_scope_name(self, name=None):
        if name is None:
            return self.__class__.__name__
        return self.__class__.__name__ + '/' + name

    def init_fn_exclude_var_names(self):
        if self._use_exclude_var_names:
            return self._init_fn_exclude_var_names
        return []

    def create_scaffold(self, saver, optimizer, ignore_global_step=True):
        exlucde_var_names = self.init_fn_exclude_var_names()
        global_step = []
        if ignore_global_step:
            global_step = ['global_step']
        exlucde_var_names.extend(
            optimizer.init_fn_exclude_var_names() + global_step)
        return saver.create_scaffold(exlucde_var_names)

    def _get_estimator_spec_dict(self, mode):
        # Values which have to be set for estimator spec.
        estimator_spec = {}
        estimator_spec['predictions'] = None
        estimator_spec['loss'] = None
        estimator_spec['train_op'] = None
        estimator_spec['eval_metric_ops'] = None
        estimator_spec['export_outputs'] = None
        estimator_spec['training_chief_hooks'] = []
        estimator_spec['training_hooks'] = []
        estimator_spec['scaffold'] = None
        estimator_spec['mode'] = mode
        return estimator_spec
