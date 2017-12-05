import interface

from meta_learning.backend.tensorflow.memory import adam_static
from meta_learning.backend.tensorflow.memory import gradient_descent_static
from meta_learning.backend.tensorflow.memory import momentum_static

create_from_params = interface.Interface.create_from_params
