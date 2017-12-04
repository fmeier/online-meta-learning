import numpy as np
import torch
from torch.nn import Parameter
import torch.nn as nn


class Rosenbrock(nn.Module):

    def __init__(self, x1_init, x2_init):
        super(Rosenbrock, self).__init__()
        self._training = True
        self._initial_params = np.ones((2, 1))
        self._initial_params[0, 0] = x1_init
        self._initial_params[1, 0] = x2_init
        self.x1 = Parameter(torch.Tensor(self._initial_params[0, :]))
        self.x2 = Parameter(torch.Tensor(self._initial_params[1, :]))
        self._a = 1.0
        self._b = 100.0
        return

    def train(self, is_train):
        self._training = is_train

    def forward(self, inputs=None):
        return self.rosenbrock(self.x1, self.x2)

    def rosenbrock(self, x1, x2):
        return (torch.pow(self._a - x1, 2) +
                self._b * torch.pow(x2 - torch.pow(x1, 2), 2))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def reset(self):
        self.x1.data[0] = self._initial_params[0, 0]
        self.x2.data[0] = self._initial_params[1, 0]
