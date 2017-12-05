
import numpy as np
import tensorflow as tf
from tf_utils import device as _device


def pairwise_distance(features, scale, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='pairwise_distance',
                           values=[features]):
        # We only support flat features.
        features.get_shape().assert_has_rank(2)
        feature_shape = tf.shape(features)
        n_samples = feature_shape[0]
        n_features = feature_shape[1]

        if scale.get_shape().ndims == 0:
            # We only have a scalar scale.
            scaled_feats = tf.scalar_mul(tf.reciprocal(scale), features)
        else:
            # For e.g. ard we have a scale per n_feature.
            with tf.control_dependencies([
                    tf.assert_equal(tf.shape(scale)[0], n_features)]):
                scaled_feats = tf.multiply(tf.reciprocal(scale), features)

        pair_1 = tf.tile(scaled_feats, [1, n_samples])
        pair_1 = tf.reshape(pair_1, [n_samples, n_samples, n_features])

        pair_2 = tf.tile(scaled_feats, [n_samples, 1])
        pair_2 = tf.reshape(pair_2, [n_samples, n_samples, n_features])

        pairwise_features = tf.square(pair_1 - pair_2)
        return tf.reduce_sum(pairwise_features, axis=2)


def kernel_squared_exponential(features, scale, sigma_f, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='kernel_squared_exponential',
                           values=[features, scale, sigma_f]):
        pair_distance = pairwise_distance(features, scale)
        sigma_f_sq = tf.square(sigma_f)
        return tf.multiply(
            sigma_f_sq, tf.exp(tf.scalar_mul(-0.5, pair_distance)))


def negative_log_likelihood(y, L, alpha):
    with _device.maybe_device_gpu():
        N = tf.to_float(tf.shape(y)[0])
        log_diag_L = tf.reduce_sum(tf.log(tf.diag_part(L)))
        alpha_y = tf.reduce_sum(tf.multiply(alpha, y))
        nlogpi = N * tf.to_float(tf.log(2.0 * np.pi))

    nll = 2.0 * log_diag_L
    nll += alpha_y
    nll += nlogpi
    return 0.5 * nll


def mean_squared_error(y, Kxx, alpha):
    with _device.maybe_device_gpu():
        N = tf.to_float(tf.shape(y)[0])
        f_mean = tf.matmul(Kxx, alpha)
        err = tf.subtract(y, f_mean)
        errsq = tf.square(err)
        return tf.scalar_mul(1.0 / N, tf.reduce_sum(errsq))


def kernel_inference(Kxx, y, sigma_y, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='kernel_inference',
                           values=[Kxx, y, sigma_y]):
        sigma_y_sq = tf.square(sigma_y)
        n_samples = tf.shape(Kxx)[0]
        kernel_y = tf.add(Kxx, tf.eye(n_samples) * sigma_y_sq)
        with _device.maybe_device_gpu():
            L = tf.cholesky(kernel_y)
            alpha = tf.cholesky_solve(L, tf.reshape(y, [n_samples, 1]))
        inference = {
            'L': L,
            'alpha': alpha,
            'f_mean': tf.matmul(Kxx, alpha)
        }
        return inference
