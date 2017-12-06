from tf_utils import factory as _factory
from tf_utils.optimizer import interface

from meta_learning.backend.tensorflow.meta_optimizer.memory import Memory
from meta_learning.backend.tensorflow.meta_optimizer.reference import Reference

create_from_params = interface.Interface.create_from_params
