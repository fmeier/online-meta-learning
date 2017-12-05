from tf_utils import factory as _factory
from tf_utils.dataset import interface
from tf_utils.dataset import mnist

from meta_learning.backend.tensorflow.dataset import mnist_class_pair
from meta_learning.backend.tensorflow.dataset import mnist_single_batch
from meta_learning.backend.tensorflow.dataset import noop

create_from_params = interface.Interface.create_from_params
