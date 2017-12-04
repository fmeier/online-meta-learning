import numpy as np
import os
#from tf_utils import utils as _utils

def get_logspace_centers_and_sigmas(min_val, max_val, num_centers, scale):
    centers_pos = np.geomspace(1e-6, max_val, num_centers/2)
    centers_neg = -np.geomspace(1e-6, -min_val, num_centers/2)
    centers_neg = centers_neg[::-1]

    centers_pos = np.hstack([0.0, centers_pos])
    center_pos_diff = centers_pos[1:] - centers_pos[:-1]
    center_neg_diff = np.copy(center_pos_diff)
    center_neg_diff = center_neg_diff[::-1]

    centers_diff = np.hstack([center_neg_diff,
                              center_neg_diff[-1],
                              center_pos_diff])

    centers = np.hstack([centers_neg, centers_pos])
    sigmasq = (scale*centers_diff)**2

    return centers, sigmasq

def plot_local_models(centers, sigmasq, x_min, x_max):
    import matplotlib.pyplot as plt
    N = 100
    M = len(centers)
    x = np.linspace(x_min, x_max, N)

    plt.figure()
    for m in range(M):
        y = np.exp(-0.5*(centers[m] - x)**2/sigmasq[m])
        plt.plot(x, y)

    plt.show()

def get_is_pytorch_path(dir_path):
    return os.path.join(dir_path, 'PYTORCH')


def touch_pytorch(dir_path):
    _utils.touch(get_is_pytorch_path(dir_path))


def is_pytorch(dir_path):
    if os.path.exists(get_is_pytorch_path(dir_path)):
        return True
    return False


def get_is_tf_path(dir_path):
    return os.path.join(dir_path, 'TF')


def touch_tf(dir_path):
    _utils.touch(get_is_tf_path(dir_path))


def is_tf(dir_path):
    if os.path.exists(get_is_tf_path(dir_path)):
        return True
    return False


def get_is_finished_path(dir_path):
    return os.path.join(dir_path, 'FINISHED')


def touch_finished(dir_path):
    _utils.touch(get_is_finished_path(dir_path))


def is_finished(dir_path):
    if os.path.exists(get_is_finished_path(dir_path)):
        return True
    return False
