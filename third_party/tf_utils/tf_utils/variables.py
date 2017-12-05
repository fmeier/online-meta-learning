
import os
import tensorflow as tf


def get_variable_names_in_scope(scope_name=None):
    if scope_name is None:
        scope_name = tf.contrib.framework.get_name_scope()
    return {var.op.name: var for var in tf.global_variables()
            if var.op.name.startswith(scope_name)}


def remove_scope_prefix(variables, scope_prefix):
    var_dict = {}
    for var_name, var in variables.items():
        name = os.path.relpath(var_name, scope_prefix)
        var_dict[name] = var
    return var_dict
