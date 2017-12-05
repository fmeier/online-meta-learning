
import tensorflow as tf
from meta_learning.backend.tensorflow.memory import interface


class Noop(interface.Interface):

    @classmethod
    def create_from_params(cls, params, **kwargs):
        return cls(params.learning_rate)

    def __init__(self, learning_rate):
        super(Noop, self).__init__()
        self._learning_rate = learning_rate

    def create_variables(self, variable, optimizer):
        pass

    def train(self, gradient_dict):
        return {}

    def inference(self, gradient_dict):
        return {'learning_rate': tf.constant(self._learning_rate)}

    def compute_activation(self, grad_feature):
        pass

    def update_memory(self, gradient_dict, grad_data, activation):
        pass

    def _inference(self, activation):
        pass
