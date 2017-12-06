import interface

from meta_learning.backend.tensorflow.base_optimizer.adam import Adam
from meta_learning.backend.tensorflow.base_optimizer.gradient_descent import GradientDescent

create_from_params = interface.Interface.create_from_params
