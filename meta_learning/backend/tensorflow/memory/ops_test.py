import unittest
import numpy as np
import tensorflow as tf

from tf_utils import defines
from tf_utils import utils

from meta_learning_tf import ops


class TestOps(unittest.TestCase):

    def _ref_activation_rbf_1d(self, data, center, sigma, active):
        data_tiled = np.tile(data, [active, 1]).flatten()
        center = center[:active].flatten()
        sigma = sigma[:active]
        return np.exp((-0.5 / sigma) * np.square(
            data_tiled - center)).flatten()

    def _ref_multi_activation_rbf(self, data, center, sigma):
        data_tiled = np.tile(np.expand_dims(data, axis=1),
                             [1, center.shape[0]])
        center = center
        sigma = sigma
        return np.exp((-0.5 / sigma) * np.square(data_tiled - center))

    def _eval_activation_rbf_1d(self, data, center, sigma, active):
        # Note we need a new graph otherwise we always connect to the
        # same graph and redefine the same variables.
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with tf.device(defines.TF_CPU):
                data_tf = tf.constant(data)
                center_tf = tf.get_variable('center', initializer=center)
                sigma_tf = tf.get_variable('sigma', initializer=sigma)
                active_tf = tf.get_variable('active', initializer=active)

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                rbf_tf = ops.activation_rbf_1d(
                    data_tf, center_tf, sigma_tf, active_tf)

                return sess.run(rbf_tf)

    def test_activation_rbf_1d(self):

        data = np.array([2.], dtype=np.float32)
        center = np.array([0., 1., 2., 3.], dtype=np.float32)
        sigma = np.ones(4, dtype=np.float32)

        for active in range(5):
            rbf_ref = self._ref_activation_rbf_1d(data, center, sigma, active)
            rbf_tf = self._eval_activation_rbf_1d(data, center, sigma, active)
            self.assertTrue(np.allclose(rbf_ref, rbf_tf))

    def test_multi_activation_rbf(self):
        data = np.array([0., 1, 2, 3], dtype=np.float32)
        center = np.array([0., 1., 2., 3.], dtype=np.float32)
        sigma = 0.5
        rbf_ref = self._ref_multi_activation_rbf(data, center, sigma)
        rbf_tf = self._eval_multi_activation_rbf(data, center, sigma)
        self.assertTrue(np.allclose(rbf_ref, rbf_tf))

    def _eval_multi_activation_rbf(self, data, center, sigma):
        # Note we need a new graph otherwise we always connect to the
        # same graph and redefine the same variables.
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with tf.device(defines.TF_CPU):
                data_tf = tf.constant(data)
                center_tf = tf.get_variable('center', initializer=center)
                sigma_tf = tf.constant(sigma)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                rbf_tf = ops.multi_activation_rbf(
                    data_tf, center_tf, sigma_tf)
                return sess.run(rbf_tf)

    def _ref_activation_update_freq(self, activation, active, threshold,
                                    activation_used,
                                    activation_count):
        activation_count = np.copy(activation_count)
        activation_used = np.copy(activation_used)
        activation_count[:active] += 1
        activation_used[:active][activation[:] >= threshold] += 1
        return activation_used, activation_count

    def _eval_activation_used_freq(self, activation, active, threshold,
                                   activation_used, activation_count):
        # Note we need a new graph otherwise we always connect to the
        # same graph and redefine the same variables.
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with tf.device(defines.TF_CPU):
                activation_tf = tf.constant(activation)
                active_tf = tf.get_variable('active', initializer=active)
                threshold_tf = tf.constant(threshold)
                activation_used_tf = tf.get_variable(
                    'activation_used', initializer=activation_used)
                activation_count_tf = tf.get_variable(
                    'activatino_count', initializer=activation_count)

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                (activation_used_tf,
                 activation_count_tf) = ops.activation_update_freq(
                    activation_tf, active_tf, threshold_tf, activation_used_tf,
                    activation_count_tf)

                return sess.run([activation_used_tf, activation_count_tf])

    def test_activation_update_freq(self):
        activation = np.array([0., 0.1, 0.2, 0.0], dtype=np.float32)
        threshold = 0.15
        activation_count = np.ones(activation.shape[0], dtype=np.int32)
        activation_used = np.ones(activation.shape[0], dtype=np.int32)

        for active in range(1, 5):
            activation_count = np.ones(activation.shape[0], dtype=np.int32)
            activation_used = np.ones(activation.shape[0], dtype=np.int32)
            activation_used_tf, activation_count_tf = (
                self._eval_activation_used_freq(
                    activation[:active], active, threshold,
                    activation_used, activation_count))
            activation_used, activation_count = (
                self._ref_activation_update_freq(
                    activation[:active], active, threshold, activation_used,
                    activation_count))
            self.assertTrue(np.allclose(activation_used, activation_used_tf))
            self.assertTrue(np.allclose(activation_count, activation_count_tf))

    def _eval_compute_eta(self, activation, value, active):
        # Note we need a new graph otherwise we always connect to the
        # same graph and redefine the same variables.
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with tf.device(defines.TF_CPU):
                learning_rate = 1.0
                activation_tf = tf.constant(activation)
                value_tf = tf.get_variable('value', initializer=value)
                active_tf = tf.get_variable('active', initializer=active)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                eta_tf = ops.compute_eta(learning_rate,
                                         activation_tf,
                                         value_tf,
                                         active_tf)
                return sess.run([eta_tf])[0]

    def _ref_compute_eta(self, activation, value, active):
        value_scaled = np.multiply(activation, value[:active])
        activation_sum = np.sum(activation)
        if activation_sum > 0:
            return np.sum(value_scaled) / np.sum(activation)
        return 0.0

    def test_compute_eta(self):
        activation = np.array([0., 0.1, 0.2, 0.0], dtype=np.float32)
        value = np.array([1., 2.0, 4.0], dtype=np.float32)

        for active in range(1, 5):
            eta = self._ref_compute_eta(activation[:active], value, active)
            eta_tf = self._eval_compute_eta(activation[:active], value, active)
            self.assertTrue(np.allclose(eta, eta_tf))

    def _ref_multi_compute_eta(self, activation, value):
        value_scaled = np.multiply(
            activation,
            np.tile(np.expand_dims(value, axis=0), [activation.shape[0], 1]))
        value_sum = np.sum(value_scaled, axis=1)
        activation_sum = np.sum(activation, axis=1)
        result = np.zeros(value_sum.shape)
        indices = np.where(activation_sum > 0)[0]
        result[indices] = value_sum[indices] / activation_sum
        return result

    def _eval_multi_compute_eta(self, activation, value):
        # Note we need a new graph otherwise we always connect to the
        # same graph and redefine the same variables.
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with tf.device(defines.TF_CPU):
                activation_tf = tf.constant(activation)
                value_tf = tf.get_variable('value', initializer=value)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                eta_tf = ops.multi_compute_eta(activation_tf,
                                               value_tf)
                return sess.run([eta_tf])[0]

    def test_multi_compute_eta(self):
        data = np.array([0., 1, 2, 3], dtype=np.float32)
        center = np.random.random_sample(20).astype(np.float32)
        sigma = 0.5
        activation = self._ref_multi_activation_rbf(data, center, sigma)
        value = np.random.random_sample(center.shape).astype(np.float32)
        ref_eta = self._ref_multi_compute_eta(
            activation=activation, value=value)
        eta = self._eval_multi_compute_eta(
            activation=activation, value=value)
        self.assertTrue(np.allclose(eta, ref_eta))

    def _ref_multi_gradient_sign_by_prod_ref(self,
                                             grad_cur,
                                             grad_prev,
                                             activation,
                                             barrier=1.):
        grad_mul = np.multiply(grad_cur, grad_prev)
        grad_mul[grad_mul > barrier] = barrier
        grad_mul[grad_mul < -barrier] = -barrier
        grad_tiled = np.tile(np.expand_dims(grad_mul, axis=1), [
                             1, activation.shape[1]])
        grad_tiled_activation = np.multiply(grad_tiled, activation)
        grad_tiled_sum = grad_tiled_activation.sum(axis=0)
        activation_sum = activation.sum(axis=0)
        result = np.zeros_like(activation_sum)
        indices = np.where(activation_sum > 0)[0]
        result[indices] = grad_tiled_sum[indices] / activation_sum[indices]
        return result

    def _eval_multi_gradient_sign_by_prod_ref(self,
                                              grad_cur,
                                              grad_prev,
                                              activation):
        # Note we need a new graph otherwise we always connect to the
        # same graph and redefine the same variables.
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with tf.device(defines.TF_CPU):
                activation_tf = tf.constant(activation)
                grad_cur_tf = tf.constant(grad_cur)
                grad_prev_tf = tf.constant(grad_prev)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                eta_tf = ops.gradient_sign_by_prod(grad_cur_tf,
                                                   grad_prev_tf,
                                                   activation_tf)
                return sess.run([eta_tf])[0]

    def test_multi_gradient_sign_by_prod_ref(self):
        grad_prev = np.array([1., 2, 0.1, 3], dtype=np.float32)
        grad_cur = np.array([-0.5, 2, -0.5, 1], dtype=np.float32)
        center = np.linspace(-5, 5, 10).astype(np.float32)
        sigma = (center[0] - center[1] * 0.5)**2
        activation = self._ref_multi_activation_rbf(grad_cur, center, sigma)
        grad_change = self._ref_multi_gradient_sign_by_prod_ref(
            grad_cur, grad_prev, activation)
        grad_change_tf = self._eval_multi_gradient_sign_by_prod_ref(
            grad_cur, grad_prev, activation)
        self.assertTrue(np.allclose(grad_change, grad_change_tf))

    def _eval_update_memory(self,
                            activation,
                            activation_used,
                            activation_count,
                            grad_cur,
                            grad_prev,
                            value,
                            center,
                            sigma,
                            active,
                            threshold_activation,
                            update_beta,
                            sigma_init):
        # Note we need a new graph otherwise we always connect to the
        # same graph and redefine the same variables.
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with tf.device(defines.TF_CPU):
                activation_tf = tf.constant(activation)
                activation_used_tf = tf.get_variable(
                    'activation_used',
                    initializer=activation_used)
                activation_count_tf = tf.get_variable(
                    'activation_count',
                    initializer=activation_count)
                grad_cur_tf = tf.constant(grad_cur)
                grad_prev_tf = tf.constant(grad_prev)
                value_tf = tf.get_variable(
                    'value',
                    initializer=value)
                center_tf = tf.get_variable(
                    'center',
                    initializer=center)
                sigma_tf = tf.get_variable(
                    'sigma',
                    initializer=sigma)
                active_tf = tf.get_variable('active', initializer=active)
                threshold_activation_tf = tf.constant(threshold_activation)
                update_beta_tf = tf.constant(update_beta)
                sigma_init_tf = tf.constant(sigma_init)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                result_op = ops.update_memory(
                    activation_tf,
                    activation_used_tf,
                    activation_count_tf,
                    grad_cur_tf,
                    grad_prev_tf,
                    value_tf,
                    center_tf,
                    sigma_tf,
                    active_tf,
                    threshold_activation_tf,
                    update_beta_tf,
                    sigma_init_tf)
                return sess.run(result_op)

    def _ref_update_memory(self,
                           activation,
                           activation_used,
                           activation_count,
                           grad_cur,
                           grad_prev,
                           value,
                           center,
                           sigma,
                           active,
                           threshold_activation,
                           update_beta,
                           sigma_init):

        barrier_max = 1.0
        barrier_min = -barrier_max

        sign_f = np.minimum(
            barrier_max,
            np.maximum(
                barrier_min,
                np.sum(np.multiply(grad_cur, grad_prev))))

        grad_prev_norm = np.linalg.norm(grad_prev)
        if active == 0:
            # We have to fill in memory.
            center[active] = grad_prev_norm
            sigma[active] = sigma_init
            value[active] = sign_f
            activation_used[active] = 1
            activation_count[active] = 1
            active += 1
            return [active, center, value, sigma, activation_used,
                    activation_count]

        activation_argmax = np.argmax(activation)
        activation_max = activation[activation_argmax]

        if activation_max < threshold_activation:
            # We have no good activation
            if active < center.shape[0]:
                # We have to fill in memory.
                center[active] = grad_prev_norm
                sigma[active] = sigma_init
                value[active] = sign_f
                activation_used[active] = 1
                activation_count[active] = 1
                active += 1
                return [active, center, value, sigma, activation_used,
                        activation_count]
            else:
                # We update the least used model.
                used_freq = np.divide(activation_used, activation_count)
                used_argmin = np.argmin(used_freq)
                center[used_argmin] = grad_prev_norm
                sigma[used_argmin] = sigma_init
                value[used_argmin] = sign_f
                activation_used[used_argmin] = 1
                activation_count[used_argmin] = 1
                return [active, center, value, sigma, activation_used,
                        activation_count]

        # We have a good activation update it.
        value_delta = sign_f * activation_max
        value[activation_argmax] += update_beta * value_delta
        return [active, center, value, sigma, activation_used,
                activation_count]

    def _data_update_memory(self, active, threshold_activation):
        np.random.seed(1234)
        result = {}
        result['activation'] = np.array([0., 0.1, 0.2, 0.0], dtype=np.float32)
        result['activation_used'] = np.ones(4, dtype=np.int32)
        result['activation_count'] = np.ones(4, dtype=np.int32)
        result['grad_cur'] = np.random.random_sample(3).astype(np.float32)
        result['grad_prev'] = np.random.random_sample(3).astype(np.float32)

        result['value'] = np.zeros(4, dtype=np.float32)
        result['center'] = np.zeros(4, dtype=np.float32)
        result['sigma'] = np.zeros(4, dtype=np.float32)

        result['active'] = active
        result['activation'] = result['activation'][:active]

        result['threshold_activation'] = threshold_activation
        result['update_beta'] = 0.9
        result['sigma_init'] = 2.0
        return result

    def test_update_memory(self):

        active = 0
        threshold_activation = 1.0
        # Test empty.
        data = self._data_update_memory(active, threshold_activation)
        (active_ref, center_ref, value_ref,
         sigma_ref, activation_used_ref, activation_count_ref) = (
             self._ref_update_memory(**data))
        data = self._data_update_memory(active, threshold_activation)
        (active_tf, center_tf, value_tf,
         sigma_tf, activation_used_tf, activation_count_tf) = (
             self._eval_update_memory(**data))
        self.assertTrue(np.allclose(active_tf, active_ref))
        self.assertTrue(np.allclose(center_tf, center_ref))
        self.assertTrue(np.allclose(value_tf, value_ref))
        self.assertTrue(np.allclose(sigma_tf, sigma_ref))
        self.assertTrue(np.allclose(activation_used_tf, activation_used_ref))
        self.assertTrue(np.allclose(activation_count_tf, activation_count_ref))

        # Test fill but full.
        active = 4
        threshold_activation = 1000.0
        data = self._data_update_memory(active, threshold_activation)
        (active_ref, center_ref, value_ref,
         sigma_ref, activation_used_ref, activation_count_ref) = (
             self._ref_update_memory(**data))
        data = self._data_update_memory(active, threshold_activation)
        (active_tf, center_tf, value_tf,
         sigma_tf, activation_used_tf, activation_count_tf) = (
             self._eval_update_memory(**data))
        self.assertTrue(np.allclose(active_tf, active_ref))
        self.assertTrue(np.allclose(center_tf, center_ref))
        self.assertTrue(np.allclose(value_tf, value_ref))
        self.assertTrue(np.allclose(sigma_tf, sigma_ref))
        self.assertTrue(np.allclose(activation_used_tf, activation_used_ref))
        self.assertTrue(np.allclose(activation_count_tf, activation_count_ref))

        # # Test fill but not full.
        active = 2
        threshold_activation = 1000.0
        data = self._data_update_memory(active, threshold_activation)
        (active_ref, center_ref, value_ref,
         sigma_ref, activation_used_ref, activation_count_ref) = (
             self._ref_update_memory(**data))
        data = self._data_update_memory(active, threshold_activation)
        (active_tf, center_tf, value_tf,
         sigma_tf, activation_used_tf, activation_count_tf) = (
             self._eval_update_memory(**data))
        self.assertTrue(np.allclose(active_tf, active_ref))
        self.assertTrue(np.allclose(center_tf, center_ref))
        self.assertTrue(np.allclose(value_tf, value_ref))
        self.assertTrue(np.allclose(sigma_tf, sigma_ref))
        self.assertTrue(np.allclose(activation_used_tf, activation_used_ref))
        self.assertTrue(np.allclose(activation_count_tf, activation_count_ref))

        # # Test update but not full.
        active = 2
        threshold_activation = 0.01
        data = self._data_update_memory(active, threshold_activation)
        (active_ref, center_ref, value_ref,
         sigma_ref, activation_used_ref, activation_count_ref) = (
             self._ref_update_memory(**data))
        data = self._data_update_memory(active, threshold_activation)
        (active_tf, center_tf, value_tf,
         sigma_tf, activation_used_tf, activation_count_tf) = (
             self._eval_update_memory(**data))
        self.assertTrue(np.allclose(active_tf, active_ref))
        self.assertTrue(np.allclose(center_tf, center_ref))
        self.assertTrue(np.allclose(value_tf, value_ref))
        self.assertTrue(np.allclose(sigma_tf, sigma_ref))
        self.assertTrue(np.allclose(activation_used_tf, activation_used_ref))
        self.assertTrue(np.allclose(activation_count_tf, activation_count_ref))

        # Test update but full.
        active = 4
        threshold_activation = 0.01
        data = self._data_update_memory(active, threshold_activation)
        (active_ref, center_ref, value_ref,
         sigma_ref, activation_used_ref, activation_count_ref) = (
             self._ref_update_memory(**data))
        data = self._data_update_memory(active, threshold_activation)
        (active_tf, center_tf, value_tf,
         sigma_tf, activation_used_tf, activation_count_tf) = (
             self._eval_update_memory(**data))
        self.assertTrue(np.allclose(active_tf, active_ref))
        self.assertTrue(np.allclose(center_tf, center_ref))
        self.assertTrue(np.allclose(value_tf, value_ref))
        self.assertTrue(np.allclose(sigma_tf, sigma_ref))
        self.assertTrue(np.allclose(activation_used_tf, activation_used_ref))
        self.assertTrue(np.allclose(activation_count_tf, activation_count_ref))


if __name__ == '__main__':
    unittest.main()
