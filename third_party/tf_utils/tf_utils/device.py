
import tensorflow as tf

from tf_utils import defines

USE_DEVICE = defines.DEVICE_CPU


def init_device():
    pass


def maybe_device_gpu(device_index=0):
    if USE_DEVICE == defines.DEVICE_GPU:
        if not tf.test.is_built_with_cuda():
            print('WARNING: Tensorflow was not built with cuda, '
                  'we use cpu mode.')
            return defines.DEVICE_CPU
        if not tf.test.is_gpu_available():
            print('WARNING: There is no GPU available we use cpu mode.')
            return defines.DEVICE_CPU
        return tf.device(
            tf.DeviceSpec(device_type=defines.DEVICE_GPU,
                          device_index=device_index))
    return tf.device(
        tf.DeviceSpec(device_type=defines.DEVICE_CPU, device_index=0))


def device_cpu():
    return tf.device(
        tf.DeviceSpec(device_type=defines.DEVICE_CPU, device_index=0))


def maybe_is_tensor_gpu(tensors):
    if isinstance(tensors, list):
        tensors = tensors[0]
    if not tensors.device:
        # The tensor is not yet placed, so it is attempted to be on
        # a GPU if the want to use a gpu.
        if USE_DEVICE == defines.DEVICE_GPU:
            return True
        return False
    elif defines.DEVICE_GPU in tensors.device:
        return True
    return False


def computation_device():
    return USE_DEVICE
