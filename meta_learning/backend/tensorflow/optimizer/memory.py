import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

from tf_utils import summaries as _summaries

from meta_learning.backend.tensorflow.memory import interface as _memory
from meta_learning.backend.tensorflow.gradient import interface as _gradient
from meta_learning.backend.tensorflow.optimizer import interface as _interface


class _MemoryOptimizer(optimizer.Optimizer):
    """Optimizer which uses memory to learn how to scale gradients online.
    @@__init__
    """

    def __init__(self,
                 memory_init_fn,
                 gradient_init_fn,
                 use_locking=False,
                 name="MemoryOptimizer"):
        super(_MemoryOptimizer, self).__init__(use_locking, name)
        self._memory_init_fn = memory_init_fn
        self._gradient_init_fn = gradient_init_fn
        self._memory = {}
        self._gradient = {}

    def _create_slots(self, var_list):
        # Create slots for the previous gradient.
        for var in var_list:
            self._memory[var.name] = self._memory_init_fn()
            self._memory[var.name].create_variables(
                var, self)

            self._gradient[var.name] = self._gradient_init_fn()
            self._gradient[var.name].create_variables(
                var, self)

    def _apply_dense(self, grad, var):
        var_name = var.name.replace(':', '_')
        with tf.variable_scope('apply_dense/{}'.format(var_name)):

            memory = self._memory[var.name]
            memory.update_slots(self, var)

            gradient = self._gradient[var.name]
            gradient.update_slots(self, var)

            grad_flat = tf.reshape(grad, [-1])

            gradient_ops = gradient.update_statistics(grad_flat)

            train_ops = memory.train(gradient_ops)

            with tf.control_dependencies(train_ops):
                inference_ops = memory.inference(gradient_ops)

            grad_apply, assign_ops = gradient.compute_apply(gradient_ops,
                                                            inference_ops)

            grad_apply = tf.reshape(grad_apply, grad.get_shape())
            if assign_ops:
                with tf.control_dependencies(assign_ops):
                    update_ops = training_ops.apply_gradient_descent(
                        var,
                        math_ops.cast(1.0, var.dtype.base_dtype),
                        grad_apply,
                        use_locking=self._use_locking).op
            else:
                update_ops = training_ops.apply_gradient_descent(
                    var,
                    math_ops.cast(1.0, var.dtype.base_dtype),
                    grad_apply,
                    use_locking=self._use_locking).op
            return update_ops

    def _apply_sparse(self, grad, var):
        raise NotImplementedError()

    def init_fn_exclude_var_names(self):
        init_fn_exclude_var_names = []
        for grad in self._gradient.values():
            init_fn_exclude_var_names.extend(grad.init_fn_exclude_var_names())
        for mem in self._memory.values():
            init_fn_exclude_var_names.extend(mem.init_fn_exclude_var_names())
        return init_fn_exclude_var_names


class Memory(_interface.Interface):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(
            use_exclude_var_names=params.get('use_exclude_var_names', False),
            params=params)

    def __init__(self,
                 use_exclude_var_names=False,
                 params=None,
                 memory_init_fn=None,
                 gradient_init_fn=None):
        super(Memory, self).__init__(
            use_exclude_var_names=use_exclude_var_names)

        if memory_init_fn is None:
            def memory_init_fn_loc():
                return _memory.Interface.create_from_params(params)
            memory_init_fn = memory_init_fn_loc

        if gradient_init_fn is None:
            def gradient_init_fn_loc():
                return _gradient.Interface.create_from_params(params)
            gradient_init_fn = gradient_init_fn_loc

        self._memory_init_fn = memory_init_fn
        self._gradient_init_fn = gradient_init_fn
        self._memory_optimizer = None

    def get_optimizer_op(self, global_step):
        self._memory_optimizer = _MemoryOptimizer(
            self._memory_init_fn,
            self._gradient_init_fn)
        return self._memory_optimizer

    def _update_init_fn_exclude_var_names(self):
        """A list of names which are excluded when loading a model.

        The default None means everything is loaded, nothing is excluded.
        """
        self._init_fn_exclude_var_names = (
            self._memory_optimizer.init_fn_exclude_var_names())
