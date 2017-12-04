
import math
import torch
import abc


class Interface(object):
    """Interface for our gradient object
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Interface, self).__init__()

    @abc.abstractmethod
    def init_state(self, grad, cuda=False):
        raise NotImplementedError()

    @abc.abstractmethod
    def update_state(self, grad):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_eta(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_grad(self):
        raise NotImplementedError()


class Sgd(Interface):

    def __init__(self, lr):
        super(Sgd, self).__init__()
        self._lr = lr
        self.state = {}

    def init_state(self, grad, cuda=False):
        pass

    def reset_state(self):
        self._grad.zero_()

    def update_state(self, grad):
        self._grad = grad

    def get_grad(self):
        return self._grad

    def compute_eta(self):
        return self._lr


class Adam(Interface):

    def __init__(self, lr, eps, beta1, beta2):
        super(Adam, self).__init__()
        self._lr = lr
        self._eps = eps
        self._beta1 = beta1
        self._beta2 = beta2
        self.state = {}

    def init_state(self, grad, cuda=False):

        self.state['step'] = 0
        # Exponential moving average of gradient values
        self.state['exp_avg_gc'] = grad.clone().zero_()
        # Exponential moving average of squared gradient values
        self.state['exp_avg_sq_gc'] = grad.clone().zero_()

        self.state['exp_avg_gc_prev'] = grad.clone().zero_()
        # Exponential moving average of squared gradient values
        self.state['exp_avg_sq_gc_prev'] = grad.clone().zero_()
        if cuda:
            self.state['bias1_power'] = torch.zeros(1).cuda()
            self.state['bias2_power'] = torch.zeros(1).cuda()
        else:
            self.state['bias1_power'] = torch.zeros(1)
            self.state['bias2_power'] = torch.zeros(1)
        self.state['bias1_power'].add_(self._beta1)
        self.state['bias2_power'].add_(self._beta2)
        self._grad = grad.clone().zero_()

    def reset_state(self):
        self.state['step'] = 0
        self.state['exp_avg_gc'].zero_()
        self.state['exp_avg_sq_gc'].zero_()
        self.state['exp_avg_gc_prev'].zero_()
        self.state['exp_avg_sq_gc_prev'].zero_()
        self.state['bias1_power'].zero_()
        self.state['bias2_power'].zero_()

        self._grad.zero_()

    def update_state(self, grad):
        self.state['step'] += 1
        exp_avg_gc, exp_avg_sq_gc = self.state[
            'exp_avg_gc'], self.state['exp_avg_sq_gc']
        exp_avg_gc.mul_(self._beta1).add_(1 - self._beta1, grad)
        exp_avg_sq_gc.mul_(self._beta2).addcmul_(1 - self._beta2, grad, grad)

        denom = exp_avg_sq_gc.sqrt().add_(self._eps)
        self._grad = torch.div(exp_avg_gc, denom)
        bias_correction1 = 1 - self._beta1 ** self.state['step']
        bias_correction2 = 1 - self._beta2 ** self.state['step']
        self._lr_adam = (
            self._lr * math.sqrt(bias_correction2) / bias_correction1)

    def compute_eta(self):
        return self._lr_adam

    def get_grad(self):
        return self._grad
