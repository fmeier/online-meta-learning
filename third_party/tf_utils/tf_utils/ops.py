
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tf_utils import utils


def exp_smooth(beta, prev, cur, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='norm',
                           values=[prev, cur]):
        return tf.add(
            tf.multiply(beta, prev),
            tf.multiply(tf.constant(1.0, dtype=prev.dtype) - beta, cur))


def print_debug(tf_op,
                tensorlist,
                message,
                is_debug,
                only_if_cpu=None,
                **kwargs):
    if not is_debug:
        return tf_op
    if only_if_cpu and utils.maybe_is_tensor_gpu(tf_op):
        return tf_op
    with utils.device_cpu():
        return tf.Print(tf_op, tensorlist, message, **kwargs)


def safe_div(numerator, denominator):
    numerator = tf.to_float(numerator)
    denominator = tf.to_float(denominator)
    return tf.where(tf.equal(denominator, 0),
                    tf.zeros_like(numerator),
                    tf.div(numerator, denominator))


def print_n_steps(variable, text, n_steps, summarize=10, data=None,
                  name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default='print_n_steps',
                           values=[variable]):
        global_step = slim.get_or_create_global_step()

        if data is None:
            data = variable

        def _print():
            return tf.Print(data, [variable], text, summarize=summarize)

        def _not_print():
            return data

        with utils.device_cpu():
            return tf.cond(tf.equal(tf.mod(global_step, n_steps), 0),
                           _print, _not_print)


def print_2d_row(data, name, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default='print_2d_row',
                           values=[data]):
        data.get_shape().assert_has_rank(2)
        with utils.device_cpu():
            return tf.Print(data, [data], name,
                            summarize=data.get_shape().as_list()[1])


def apply_indices(data_dict, indices, ignore_keys=[], name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='apply_indices',
                           values=data_dict.values()):
        result = {}

        for key, value in data_dict.items():
            if key in ignore_keys:
                result[key] = value
                continue
            with tf.variable_scope('apply_indices_' + key, values=[value]):
                static_shape = value.get_shape().as_list()
                if static_shape[-1] is not None:
                    result[key] = gather_with_shape(value, indices)
                else:
                    result[key] = tf.gather(value, indices)
        return result


def gather_with_shape(data, indices, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='gather_with_shape',
                           values=[data, indices]):
        shape = data.get_shape().as_list()
        result = tf.gather(data, indices)
        return tf.reshape(result, [-1] + shape[1:])


def concat_with_shape(dimension, data_list, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='gather_with_shape',
                           values=data_list):
        shape = data_list[0].get_shape().as_list()
        result = tf.concat(axis=dimension, values=data_list)
        target_shape = shape[:dimension] + [-1] + shape[dimension + 1:]
        return tf.reshape(result, target_shape)


def binarize(tensor, condition, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='binarize',
                           values=[tensor, condition]):
        return tf.where(
            condition,
            tf.ones_like(tensor),
            tf.zeros_like(tensor))


def mask_argmax_column(data, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='mask_argmax_column',
                           values=[data]):
        n_data = tf.shape(data)[0]
        # Now we can pick the argmax for every column.
        data_argmax_column = tf.argmax(data, 0)

        # We create a one hot vector for every column.
        data_mask = tf.transpose(
            tf.one_hot(data_argmax_column, depth=n_data))

        return tf.multiply(data, data_mask)


def mask_argmax_row(data, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='mask_argmax_row',
                           values=[data]):
        n_data = tf.shape(data)[1]

        # Now we can pick the argmax for every column.
        data_argmax_row = tf.argmax(data, 1)

        # We create a one hot vector for every column.
        data_mask = tf.one_hot(data_argmax_row, depth=n_data)

        return tf.multiply(data, data_mask)
