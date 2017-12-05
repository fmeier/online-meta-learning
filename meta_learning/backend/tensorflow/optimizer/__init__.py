from tf_utils import factory as _factory
from tf_utils.optimizer import interface
from tf_utils.optimizer import adam
from tf_utils.optimizer import momentum
from tf_utils.optimizer import gradient_descent

from meta_learning.backend.tensorflow.optimizer.memory import Memory
from meta_learning.backend.tensorflow.optimizer.reference import Reference

create_from_params = interface.Interface.create_from_params
