import interface

from meta_learning.backend.tensorflow.memory.adam_static import AdamStatic
from meta_learning.backend.tensorflow.memory.gradient_descent_static import GradientDescentStatic
from meta_learning.backend.tensorflow.memory.momentum_static import MomentumStatic

create_from_params = interface.Interface.create_from_params
