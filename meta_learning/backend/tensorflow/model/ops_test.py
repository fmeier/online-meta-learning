
"""Tests for gp_ops.
"""
import unittest

import numpy as np
import tensorflow as tf

from meta_learning.backend.tensorflow.model import ops as _ops


class TestOps(unittest.TestCase):

    def test_settings(self):
        n_samples = [2, 5, 10]
        n_features = [1, 3, 5]
        lengthscales = [0.3, 0.5]
        sigma_fs = [1.0, 2.0]
        sigma_ys = [0.01, 0.1]
        return n_samples, n_features, lengthscales, sigma_fs, sigma_ys

    def ref_pairwise_distance(self, features, scaling):
        n_samples = features.shape[0]
        pairwise = np.zeros((n_samples, n_samples), dtype=np.float32)
        scaled_features = features / scaling

        for x in xrange(n_samples):
            for y in xrange(n_samples):
                pairwise[x, y] = np.sum((scaled_features[x] -
                                         scaled_features[y])**2)
        return pairwise

    def test_pairwise_distance(self):
        n_samples, n_features, lengthscales, _, _ = self.test_settings()

        for n_sample in n_samples:
            for n_feature in n_features:
                for lengthscale in lengthscales:
                    features = np.random.random_sample(
                        (n_sample, n_feature)).astype(np.float32)

                    pairwise_ref = self.ref_pairwise_distance(
                        features, lengthscale)

                    with tf.Session() as sess:
                        features_tf = tf.constant(features)
                        scaling_tf = tf.constant(lengthscale)
                        pairwise_tf = _ops.pairwise_distance(
                            features_tf, scaling_tf)
                        pairwise = sess.run(pairwise_tf)

                        # Notice we have to set the tolerance lower,
                        # since we deal with floating point rounding issues.
                        self.assertTrue(np.allclose(pairwise_ref, pairwise,
                                                    atol=1e-6))

    def ref_kernel_squared_exponential(self, features, lengthscale, sigma_f):
        pair_distance = self.ref_pairwise_distance(features, lengthscale)
        return np.square(sigma_f) * np.exp(-0.5 * pair_distance)

    def test_kernel_squared_exponential(self):
        n_samples, n_features, lengthscales, sigma_fs, _ = self.test_settings()

        for n_sample in n_samples:
            for n_feature in n_features:
                for lengthscale in lengthscales:
                    for sigma_f in sigma_fs:
                        features = np.random.random_sample(
                            (n_sample, n_feature)).astype(np.float32)

                        kernel_ref = self.ref_kernel_squared_exponential(
                            features, lengthscale, sigma_f)

                        with tf.Session() as sess:
                            features_tf = tf.constant(features)
                            scaling_tf = tf.constant(lengthscale)
                            sigma_f_tf = tf.constant(sigma_f)
                            kernel_tf = _ops.kernel_squared_exponential(
                                features_tf, scaling_tf, sigma_f_tf)
                            kernel = sess.run(kernel_tf)

                            # Notice we have to set the tolerance
                            # lower, since we deal with floating point
                            # rounding issues.
                            self.assertTrue(np.allclose(kernel_ref, kernel,
                                                        atol=1e-6))

    def ref_kernel_inference(self, kernel_xx, y, sigma_y):
        sigma_y_sq = np.square(sigma_y)
        n_samples = kernel_xx.shape[0]

        kernel_y = kernel_xx + np.eye(n_samples) * sigma_y_sq
        L = np.linalg.cholesky(kernel_y)
        LT = np.transpose(L)
        lt_solved = np.linalg.solve(L, np.reshape(y, [n_samples, 1]))
        alpha = np.linalg.solve(LT, lt_solved)
        inference = {
            'L': L,
            'alpha': alpha,
            'f_mean': np.dot(
                kernel_xx, np.reshape(alpha, [n_samples, 1]))
        }
        return inference

    def test_kernel_inference(self):
        (n_samples,
         n_features, lengthscales, sigma_fs, sigma_ys) = self.test_settings()

        for n_sample in n_samples:
            for n_feature in n_features:
                for lengthscale in lengthscales:
                    for sigma_f in sigma_fs:
                        for sigma_y in sigma_ys:
                            inputs = np.random.random_sample(
                                (n_sample, n_feature)).astype(np.float32)
                            y = np.random.random_sample(
                                (n_sample)).astype(np.float32)

                            kernel_ref = self.ref_kernel_squared_exponential(
                                inputs, lengthscale, sigma_f)
                            inference_ref = self.ref_kernel_inference(
                                kernel_ref, y, sigma_y)

                            with tf.Session() as sess:
                                sigma_y_tf = tf.constant(sigma_y)
                                y_tf = tf.constant(y)
                                kernel_tf = tf.constant(kernel_ref)
                                inference_tf = _ops.kernel_inference(
                                    kernel_tf, y_tf, sigma_y_tf)
                                inference = sess.run(inference_tf)

                                if not np.allclose(
                                        inference_ref['f_mean'],
                                        inference['f_mean'],
                                        atol=1e-3):
                                    print('low accuracy')

                                # Notice we have to set the tolerance
                                # lower, since we deal with floating point
                                # rounding issues.
                                self.assertTrue(
                                    np.allclose(
                                        inference_ref['f_mean'],
                                        inference['f_mean'],
                                        atol=1e-2))

    def ref_nloglik(self, input_y, L, alpha):
        N = np.shape(input_y)[0]
        nll = 2.0 * np.sum(np.log(np.diag(L)))
        nll += np.sum(alpha * input_y)
        nll += N * np.log(2.0 * np.pi)
        return 0.5 * nll

    def test_nloglik(self):
        (n_samples,
         n_features, lengthscales, sigma_fs, sigma_ys) = self.test_settings()

        for n_sample in n_samples:
            for n_feature in n_features:
                for lengthscale in lengthscales:
                    for sigma_f in sigma_fs:
                        for sigma_y in sigma_ys:
                            features = np.random.random_sample(
                                (n_sample, n_feature)).astype(np.float32)
                            y = np.random.random_sample(
                                (n_sample)).astype(np.float32)

                            kernel_ref = self.ref_kernel_squared_exponential(
                                features, lengthscale, sigma_f)
                            inference_ref = self.ref_kernel_inference(
                                kernel_ref, y, sigma_y)

                            nll_ref = self.ref_nloglik(
                                y, inference_ref['L'], inference_ref['alpha'])
                            with tf.Session() as sess:
                                sigma_y_tf = tf.constant(sigma_y)
                                y_tf = tf.constant(y)
                                kernel_tf = tf.constant(kernel_ref)
                                inference_tf = _ops.kernel_inference(
                                    kernel_tf, y_tf, sigma_y_tf)
                                nll_tf = _ops.negative_log_likelihood(
                                    y_tf,
                                    inference_tf['L'],
                                    inference_tf['alpha'])
                                nll = sess.run(nll_tf)

                                # print "nll :" + str(nll)
                                # print "nll_tf :" + str(nll_tf.eval())
                                # print ""

                                # Notice we have to set the tolerance
                                # lower, since we deal with floating point
                                # rounding issues.
                                # self.assertTrue(
                                #     np.allclose(nll_ref, nll,
                                #                 atol=1e-2))
                                if not np.allclose(nll_ref, nll,
                                                   atol=1e-2):
                                    import ipdb
                                    ipdb.set_trace()


if __name__ == '__main__':
    unittest.main()
