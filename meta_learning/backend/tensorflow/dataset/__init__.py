from tf_utils import factory as _factory
from tf_utils.dataset import interface
from tf_utils.dataset import mnist

from meta_learning.backend.tensorflow.dataset.mnist_class_pair import MnistClassPair
from meta_learning.backend.tensorflow.dataset.mnist_single_batch import MnistSingleBatch
from meta_learning.backend.tensorflow.dataset.noop import Noop

create_from_params = interface.Interface.create_from_params
