import torch
from torch.nn import Module
from torch.autograd import Variable


class NMSELoss(Module):
    def __init__(self, var_data):
        super(NMSELoss, self).__init__()
        self.var_data = var_data

    def forward(self, inputs, targets):
        weighting = Variable(self.var_data, requires_grad=False)
        err = inputs - targets
        return torch.mean(torch.mean(err**2, 0) / weighting)


class IdentityLoss(Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()

    def forward(self, inputs, targets):
        return inputs
