
import tensorflow as tf


def variables_list_from_checkpoint(checkpoint_path):
    variables = tf.contrib.framework.list_variables(
        checkpoint_path)
    return [var[0] for var in variables]


def variables_create_mapping(variables_in_scope, variables_in_checkpoint):
    var_dict = {}
    for variable_name in variables_in_checkpoint:
        for var_name, var in variables_in_scope.items():
            if variable_name in var_name:
                var_dict[variable_name] = var
    return var_dict


def create_init_fn(checkpoint_path,
                   exclude_var_names=None):

    if checkpoint_path is not None:
        variables_to_restore = tf.global_variables()
        print('::::::::::::::::::::::::')
        print('::variables for potential restore::')
        print('::::::::::::::::::::::::')
        for name in sorted([var.op.name for var in variables_to_restore]):
            print(name)

        if exclude_var_names is not None:
            print('::::::::::::::::::::::::')
            print('::variables to exclude::')
            print('::::::::::::::::::::::::')
            for name in sorted(exclude_var_names):
                print(name)

        variables_in_checkpoint = variables_list_from_checkpoint(
            checkpoint_path)

        print('::::::::::::::::::::::::')
        print('::variables in checkpoint::')
        print('::::::::::::::::::::::::')
        for name in sorted(variables_in_checkpoint):
            print(name)

        vars_dict = {}
        exclude = []
        if exclude_var_names is not None:
            exclude.extend(exclude_var_names)

        variables_excluded_count = 0
        for var in variables_to_restore:
            for var_name in variables_in_checkpoint:
                var_op = var.op.name
                if var_op in exclude:
                    variables_excluded_count += 1
                    continue
                if var.op.name.endswith(var_name):
                    vars_dict[var_name] = var

        print('::::::::::::::::::::::::')
        print('::variables to restore::')
        print('::::::::::::::::::::::::')
        for name in sorted(vars_dict.keys()):
            print(name)

        if variables_excluded_count < len(variables_to_restore) and (
                not vars_dict):
            raise Exception(
                'We try to load variables from checkpoint {}'
                ' but there was no matching one.'.format(checkpoint_path))

        # We have to check if we have something to restore.
        if vars_dict:
            return tf.contrib.framework.assign_from_checkpoint_fn(
                tf.train.latest_checkpoint(checkpoint_path),
                vars_dict)
    return None
