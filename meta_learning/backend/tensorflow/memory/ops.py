import tensorflow as tf
from tf_utils import ops as _ops


def norm(value, name_or_scope=None):
    with tf.variable_scope(name_or_scope,
                           default_name='norm',
                           values=[value]):
        # import ipdb
        # ipdb.set_trace()
        # value.get_shape().assert_has_rank(1)
        return tf.sqrt(tf.reduce_sum(tf.square(value)))


def gradient_sign_by_prod(grad_cur,
                          grad_prev,
                          activation,
                          barrier=1,
                          name_or_scope=None,
                          is_debug=False):
    """Returns the scaled gradients to update the memory.

    grad_sign = grad_cur * grad_prev
    We magnify the learning rate if the sign aggrees
    positive * positive -> > 0
    positive * negative -> < 0
    negative * negative -> > 0

    Args:
        grad_cur: A tensor of arbitrary shape.
        grad_prev: A tensor of arbitrary shape, same as grad_cur
        activation: centers * size(grad_cur)

    Returns:
        A tensor of size(grad_cur).
    """
    with tf.variable_scope(name_or_scope,
                           default_name='gradient_sign_ref',
                           values=[grad_cur, grad_prev, activation]):
        grad_cur.get_shape().assert_is_compatible_with(grad_prev.get_shape())
        grad_cur = tf.reshape(grad_cur, [-1])
        grad_prev = tf.reshape(grad_prev, [-1])
        assert (activation.get_shape().as_list()[0] ==
                grad_cur.get_shape().as_list()[0])

        barrier_max = tf.constant(barrier, dtype=tf.float32)
        barrier_min = -barrier_max

        grad_change_raw = tf.maximum(
            barrier_min,
            tf.minimum(
                barrier_max,
                tf.multiply(grad_cur, grad_prev)))
        grad_change_raw_tiled = tf.tile(
            tf.expand_dims(grad_change_raw, axis=1),
            [1, activation.get_shape().as_list()[1]])

        # everything is multiplied by its activation
        grad_change_activated = tf.reduce_mean(
            tf.multiply(activation, grad_change_raw_tiled), axis=0)

        # previously we basically always set it to 1 and -1 for the
        # numerical precision of a Gaussian.
        # grad_change = _ops.safe_div(grad_change_activated,
        #                             activation_sum)

        # We have to divide by the number of models which can be active.
        # actually this is dividing by the size of the gradient
        # - which makes maore sense
        # grad_change = tf.div(grad_change_activated,
        #                      activation.get_shape().as_list()[0])

        return grad_change_activated


def gradient_sign(grad_cur,
                  grad_prev,
                  activation,
                  barrier=1,
                  name_or_scope=None,
                  is_debug=False):
    """Returns the scaled gradients to update the memory.

    grad_sign = sign(grad_cur * grad_prev)

    Args:
        grad_cur: A tensor of arbitrary shape.
        grad_prev: A tensor of arbitrary shape, same as grad_cur
        activation: centers * size(grad_cur)

    Returns:
        A tensor of size(grad_cur).
    """
    with tf.variable_scope(name_or_scope,
                           default_name='gradient_sign_ref',
                           values=[grad_cur, grad_prev, activation]):
        grad_cur.get_shape().assert_is_compatible_with(grad_prev.get_shape())
        grad_cur = tf.reshape(grad_cur, [-1])
        grad_prev = tf.reshape(grad_prev, [-1])
        assert (activation.get_shape().as_list()[0] ==
                grad_cur.get_shape().as_list()[0])

        grad_change_raw = tf.sign(tf.multiply(grad_cur, grad_prev))
        grad_change_raw_tiled = tf.tile(
            tf.expand_dims(grad_change_raw, axis=1),
            [1, activation.get_shape().as_list()[1]])
        # everything is multiplied by its activation
        grad_change_activated = tf.reduce_sum(
            tf.multiply(activation, grad_change_raw_tiled), axis=0)
        activation_sum = tf.reduce_sum(activation, axis=0)
        grad_change = _ops.safe_div(grad_change_activated,
                                    activation_sum)

        return grad_change


def activation_rbf_2d(data, center, sigma, active, name_or_scope=None,
                      is_debug=False):
    """Return an activation tensor for the active centers.

    We return the rbf function for one dimensional data.
        exp(-0.5 / sigma * (data - center[:active])**2)

    Args:
        data (2): A tensor concerning the current data value.
        center (max_centers): A variable containing the center values.
        sigma (1): A variable containing the sigma scaling.
        active (0): A variable to indicate the number of active centers.

    Returns:
        A tensor of shape [active].
    """
    with tf.variable_scope(name_or_scope,
                           default_name='activation_rbf_1d',
                           values=[data, center, sigma, active]):
        data.get_shape().assert_has_rank(2)
        center.get_shape().assert_has_rank(2)
        sigma.get_shape().assert_has_rank(1)
        active.get_shape().assert_has_rank(0)

        data_tiled = tf.tile(data, [active, 1])

        center_sliced = tf.slice(center,
                                 begin=[0, 0], size=[active, -1])

        sigma_sliced = tf.slice(sigma, begin=[0], size=[active])

        sigma_scaled = tf.divide(tf.constant(-0.5), sigma_sliced)

        center_diff = tf.subtract(data_tiled, center_sliced)

        center_diff_square = tf.reduce_sum(tf.square(center_diff), axis=1)

        def _do_print():
            sigma_ = sigma_scaled
            sigma_ = _ops.print_debug(sigma_,
                                      [active],
                                      "mth_rbf",
                                      is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_, [sigma_[active - 1]],
                                      "rbf_sigma", is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_, [data_tiled[active - 1]],
                                      "rbf_x", is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_, [center_sliced[active - 1]],
                                      "rbf_c", is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_, [center_diff[active - 1]],
                                      "rbf_x_minus_c", is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_,
                                      [center_diff_square[active - 1]],
                                      "rbf_centerdiff", is_debug=is_debug)
            return sigma_

        def _dont_print():
            return sigma_scaled

        sigma_scaled = tf.cond(tf.equal(active, 0), _dont_print, _do_print)
        return tf.exp(tf.multiply(sigma_scaled, center_diff_square))


def multi_activation_rbf(data,
                         center,
                         sigma,
                         name_or_scope=None,
                         is_debug=False):
    """Return an activation tensor for the active centers.

    We return the rbf function for one dimensional data.
        exp(-0.5 / sigma * (data - center[:active])**2)

    Args:
        data (1): A tensor concerning the current data value.
        center (max_centers): A variable containing the center values.
        sigma: A scalar constant containing for the sigma scaling.

    Returns:
        A tensor of shape [active].
    """
    with tf.variable_scope(name_or_scope,
                           default_name='multi_activation_rbf',
                           values=[data, center, ]):
        data.get_shape().assert_has_rank(1)
        center.get_shape().assert_has_rank(1)

        data_tiled = tf.tile(tf.expand_dims(data, axis=1),
                             [1, center.get_shape().as_list()[0]],
                             name='tile_data_to_center')
        center_diff = tf.subtract(data_tiled, center, name='d_minus_c')
        center_diff_square = tf.square(center_diff, name='d_minus_c_squared')
        sigma_scaled = tf.divide(tf.constant(-0.5), sigma)

        return tf.exp(tf.multiply(
            sigma_scaled, center_diff_square, name='sigma_times_mu'))


def activation_rbf_1d(data,
                      center,
                      sigma,
                      active,
                      name_or_scope=None,
                      is_debug=False):
    """Return an activation tensor for the active centers.

    We return the rbf function for one dimensional data.
        exp(-0.5 / sigma * (data - center[:active])**2)

    Args:
        data (1): A tensor concerning the current data value.
        center (max_centers): A variable containing the center values.
        sigma (1): A variable containing the sigma scaling.
        active (0): A variable to indicate the number of active centers.

    Returns:
        A tensor of shape [active].
    """
    with tf.variable_scope(name_or_scope,
                           default_name='activation_rbf_1d',
                           values=[data, center, sigma, active]):
        data.get_shape().assert_has_rank(1)
        center.get_shape().assert_has_rank(1)
        sigma.get_shape().assert_has_rank(1)
        active.get_shape().assert_has_rank(0)

        data_tiled = tf.tile(data, [active])

        center_sliced = tf.slice(center,
                                 begin=[0], size=[active])

        sigma_sliced = tf.slice(sigma, begin=[0], size=[active])

        sigma_scaled = tf.divide(tf.constant(-0.5), sigma_sliced)

        center_diff = tf.subtract(data_tiled, center_sliced)

        center_diff_square = tf.square(center_diff)

        def _do_print():
            sigma_ = sigma_scaled
            sigma_ = _ops.print_debug(sigma_,
                                      [active],
                                      "mth_rbf",
                                      is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_, [sigma_[active - 1]],
                                      "rbf_sigma", is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_, [data_tiled[active - 1]],
                                      "rbf_x", is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_, [center_sliced[active - 1]],
                                      "rbf_c", is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_, [center_diff[active - 1]],
                                      "rbf_x_minus_c", is_debug=is_debug)
            sigma_ = _ops.print_debug(sigma_,
                                      [center_diff_square[active - 1]],
                                      "rbf_centerdiff", is_debug=is_debug)
            return sigma_

        def _dont_print():
            return sigma_scaled

        sigma_scaled = tf.cond(tf.equal(active, 0), _dont_print, _do_print)
        return tf.exp(tf.multiply(sigma_scaled, center_diff_square))


def activation_update_freq(activation,
                           active,
                           threshold,
                           activation_used,
                           activation_count,
                           name_or_scope=None):
    """Update the variables _used and _count with the latest activations.

    If an activation is above a threshold we increment the
    activation_used count and always the activation_cout.
    We only do this for the parts of the actiations which are already
    active meaning we have set the centers.

    Args:
        activation (active): A tensor with the activations.
        active (0): A variable with the number of active centers.
        threshold (0): A tensor for >= threshold.
        activation_used (max_centers): A variable containing the used count.
        activation_count (max_centers): A variable continanit the total count.

    Returns:
        The assigned activation_used and activation_count.
    """
    with tf.variable_scope(name_or_scope,
                           default_name='activation_update_active',
                           values=[activation,
                                   active,
                                   threshold,
                                   activation_used,
                                   activation_count]):
        activation.get_shape().assert_has_rank(1)
        active.get_shape().assert_has_rank(0)
        activation_used.get_shape().assert_has_rank(1)
        activation_count.get_shape().assert_has_rank(1)

        # We use the fact that true = 1.0 and false 0.0
        update_used_sliced = tf.to_int32(
            tf.greater_equal(activation, threshold))

        elements_unused = tf.shape(activation_used)[0] - active
        update_used = tf.concat(
            0, [update_used_sliced, tf.zeros([elements_unused],
                                             dtype=tf.int32)])
        activation_used = activation_used.assign_add(update_used)

        update_count = tf.concat(
            0, [tf.ones([active], dtype=tf.int32),
                tf.zeros([elements_unused], dtype=tf.int32)])
        activation_count = activation_count.assign_add(update_count)
        return activation_used, activation_count


def multi_compute_eta(activation,
                      value,
                      name_or_scope=None,
                      is_debug=False):
    """Return the new scaling for the learning rate.


    Args:
        activation (active): A tensor with the activations.
        value (max_centers): A variable containing the value for the center.
        active (0): A variable with the number of active centers.

    Returns:
        A tensor of shape [active, 1].
    """
    with tf.variable_scope(name_or_scope,
                           default_name='compute_eta',
                           values=[activation, value]):
        assert (activation.get_shape().as_list()[1] ==
                value.get_shape().as_list()[0])

        activation_total = tf.reduce_sum(activation, axis=1)
        value_tiled = tf.tile(
            tf.expand_dims(value, axis=0),
            [activation.get_shape().as_list()[0], 1])
        activated_value = tf.reduce_sum(
            tf.multiply(value_tiled, activation), axis=1)
        return _ops.safe_div(activated_value, activation_total)


def compute_eta(learning_rate,
                activation,
                value,
                active,
                name_or_scope=None,
                is_debug=False):
    """Return the new scaling for the learning rate.


    Args:
        activation (active): A tensor with the activations.
        value (max_centers): A variable containing the value for the center.
        active (0): A variable with the number of active centers.

    Returns:
        A tensor of shape [active, 1].
    """
    with tf.variable_scope(name_or_scope,
                           default_name='compute_eta',
                           values=[activation, value, active]):
        activation.get_shape().assert_has_rank(1)
        active.get_shape().assert_has_rank(0)

        def _empty():
            return tf.constant(learning_rate)

        def _eta():
            active_ = _ops.print_debug(
                active, [active], 'compute_eta_active', is_debug=is_debug)
            value_sliced = tf.slice(value, begin=[0], size=[active_])
            activation_ = _ops.print_debug(
                activation,
                [activation[active_ - 1], value_sliced],
                'compute_eta_activation_value',
                is_debug=is_debug)
            eta = tf.multiply(value_sliced, activation_)
            activation_total = tf.reduce_sum(activation)
            return tf.cond(
                tf.equal(activation_total, 0),
                _empty,
                lambda: _ops.safe_div(tf.reduce_sum(eta), activation_total))

        return tf.cond(tf.equal(active, 0), _empty, _eta)


def update_memory(activation,
                  activation_used,
                  activation_count,
                  grad_cur,
                  grad_prev,
                  grad_data,
                  value,
                  center,
                  sigma,
                  active,
                  threshold_activation,
                  update_beta,
                  sigma_init,
                  name_or_scope=None,
                  is_debug=False):
    """Update the memory by either creating a new center or updating a one.
    """
    with tf.variable_scope(name_or_scope,
                           default_name='update_memory',
                           values=[activation, grad_cur, grad_prev, value,
                                   center, sigma, active, threshold_activation,
                                   update_beta, sigma_init]):
        activation.get_shape().assert_has_rank(1)
        activation_used.get_shape().assert_has_rank(1)
        activation_count.get_shape().assert_has_rank(1)
        grad_prev = _ops.print_debug(grad_prev,
                                     [update_beta, grad_cur, grad_prev],
                                     "update_memory_beta_grad_cur_grad_prev",
                                     is_debug=is_debug)
        tf.summary.scalar('update_memory_active', active)
        # tf.summary.scalar('update_memory_grad_cur', grad_cur)
        # Franzi: apparently the gradients have rank 0
        # grad_cur.get_shape().assert_has_rank(0)
        # grad_prev.get_shape().assert_has_rank(0)
        # center.get_shape().assert_has_rank(1)
        sigma.get_shape().assert_has_rank(1)
        active.get_shape().assert_has_rank(0)
        barrier_max = tf.constant(1, dtype=tf.float32)
        barrier_min = -barrier_max

        grad_dotproduct = tf.reduce_sum(tf.multiply(grad_prev, grad_cur))
        grad_data_shape = grad_data.get_shape().as_list()
        sign_f = tf.minimum(barrier_max, tf.maximum(
            barrier_min, grad_dotproduct))
        sign_f = _ops.print_debug(
            sign_f, [sign_f, grad_dotproduct], 'sign_f, grad_dotproduct',
            is_debug=is_debug)
        tf.summary.scalar('sign_f', sign_f)

        def _add_center_empty():
            with tf.variable_scope('_add_center_free'):
                index = active
                active_update = tf.add(active, tf.constant(1, dtype=tf.int32))
                activation_used_update = tf.constant([1], dtype=tf.int32)
                activation_count_update = tf.constant([1], dtype=tf.int32)
                # we should use the value of the closest neighbor
                value_update = tf.reshape(update_beta, [1])
                center_update = grad_data
                sigma_update = tf.reshape(sigma_init, [1])
                return (
                    index,
                    active_update,
                    activation_used_update,
                    activation_count_update,
                    value_update,
                    center_update,
                    sigma_update)

        def _add_center_free():
            with tf.variable_scope('_add_center_free'):
                index = active
                # pick the index with largest activation
                index_closest = tf.to_int32(tf.argmax(activation, axis=0))

                active_update = tf.add(active, tf.constant(1, dtype=tf.int32))
                activation_used_update = tf.constant([1], dtype=tf.int32)
                activation_count_update = tf.constant([1], dtype=tf.int32)
                # we should use the value of the closest neighbor
                value_update = tf.reshape(value[index_closest], [1])
                center_update = grad_data
                sigma_update = tf.reshape(sigma_init, [1])
                return (
                    index,
                    active_update,
                    activation_used_update,
                    activation_count_update,
                    value_update,
                    center_update,
                    sigma_update)

        def _add_center_full():
            with tf.variable_scope('_add_center_full'):

                used_freq = _ops.safe_div(
                    activation_used, activation_count)

                index = tf.to_int32(tf.argmin(used_freq, axis=0))
                index = _ops.print_debug(index, [index], 'add_center_full',
                                         is_debug=is_debug)

                active_update = active

                activation_used_update = tf.constant([1], dtype=tf.int32)
                activation_count_update = tf.constant([1], dtype=tf.int32)

                value_update = tf.reshape(update_beta, [1])
                center_update = grad_data
                sigma_update = tf.reshape(sigma_init, [1])
                return (
                    index,
                    active_update,
                    activation_used_update,
                    activation_count_update,
                    value_update,
                    center_update,
                    sigma_update)

        def _update_center():
            with tf.variable_scope('_update_center'):
                index = tf.to_int32(tf.argmax(activation, axis=0))

                index = _ops.print_debug(index, [index], 'update_center',
                                         is_debug=is_debug)
                active_update = active

                activation_used_update = tf.reshape(
                    activation_used[index], [1])
                activation_count_update = tf.reshape(
                    activation_count[index], [1])

                activation_max = activation[index]
                activation_max = _ops.print_debug(
                    activation_max,
                    [value[index], update_beta, activation_max],
                    'update_center_value_beta_activation_max',
                    is_debug=is_debug)
                value_update = tf.reshape(
                    tf.add(value[index],
                           tf.multiply(update_beta,
                                       tf.multiply(sign_f,
                                                   activation_max))), [1])
                value_update = _ops.print_debug(
                    value_update, [value_update, sign_f],
                    'update_center_value_updata',
                    is_debug=is_debug)
                center_update = tf.reshape(center[index], grad_data_shape)
                sigma_update = tf.reshape(sigma[index], [1])
                return (
                    index,
                    active_update,
                    activation_used_update,
                    activation_count_update,
                    value_update,
                    center_update,
                    sigma_update)

        def _add_center_free_or_full():
            with tf.variable_scope('_add_center_free_or_full'):
                # If all centers are filled we have to update one
                # otherwise just add a center.
                return tf.cond(tf.less(active, tf.shape(center)[0]),
                               _add_center_free,
                               _add_center_full)

        def _add_or_update():
            with tf.variable_scope('_add_or_update'):
                activation_max = tf.reduce_max(activation, axis=0)
                activation_max = _ops.print_debug(
                    activation_max, [activation_max],
                    'add_or_update_activation_max',
                    is_debug=is_debug)
                return tf.cond(tf.less(activation_max,
                                       threshold_activation),
                               _add_center_free_or_full,
                               _update_center)

        (index,
         active_update,
         activation_used_update,
         activation_count_update,
         value_update,
         center_update,
         sigma_update) = tf.cond(tf.equal(active, 0),
                                 _add_center_empty,
                                 _add_or_update)

        index = _ops.print_debug(index, [index], 'update_memory_index',
                                 is_debug=is_debug)
        activation_used_new = tf.scatter_update(
            activation_used, [index], activation_used_update,
            name='activation_used')
        activation_count_new = tf.scatter_update(
            activation_count, [index], activation_count_update,
            name='activation_count')
        # value_new = tf.scatter_update(
        #     value, [index], tf.maximum(value_update, 0.0001), name='value')
        value_new = tf.scatter_update(
            value, [index], value_update, name='value')
        center_new = tf.scatter_update(
            center, [index], center_update, name='center')
        sigma_new = tf.scatter_update(
            sigma, [index], sigma_update, name='sigma')
        with tf.control_dependencies(
                [activation_used_new, activation_count_new, value_new,
                 center_new, sigma_new]):
            active_new = tf.assign(
                active, active_update, name='active')
            return (active_new, center_new, value_new,
                    sigma_new, activation_used_new,
                    activation_count_new)
