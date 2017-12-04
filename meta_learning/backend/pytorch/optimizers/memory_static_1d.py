import torch
from meta_learning.backend.pytorch.optimizers import memory
from meta_learning.utils import get_logspace_centers_and_sigmas
# import ipdb


class MemoryStatic1d(memory.Interface):

    def __init__(self,
                 meta_params,
                 learning_rate_init,
                 memory_optimizer_init_fn,
                 is_debug):
        super(MemoryStatic1d, self).__init__()

        self.num_centers = meta_params['num_centers']
        self.max_grad_value = meta_params['max_grad_value']
        self.min_grad_value = meta_params['min_grad_value']

        lm_scale = meta_params['lm_scale']

        if meta_params['logspace']:
            (centers,
             sigmasq) = get_logspace_centers_and_sigmas(self.min_grad_value,
                                                        self.max_grad_value,
                                                        self.num_centers,
                                                        lm_scale)
            self.centers = torch.Tensor(centers)
            self.sigmasq = torch.Tensor(sigmasq)
            self.num_centers += 1
        else:
            self.centers = torch.linspace(self.min_grad_value,
                                          self.max_grad_value,
                                          self.num_centers)
            self.sigmasq = (lm_scale * (self.centers[0] - self.centers[1]))**2
            self.sigmasq = torch.Tensor([self.sigmasq])

        self.value = torch.ones(self.num_centers) * learning_rate_init
        self.memory_learning_rate = meta_params['memory_learning_rate']
        self.memory_optimizer = memory_optimizer_init_fn()

        self.is_debug = is_debug
        self.state = {}
        self.num_local_models = self.num_centers

    def compute_grad_change(self):
        grad_change_raw = self.state['grad'] * self.state['grad_prev']
        grad_change = torch.max(
            torch.min(grad_change_raw, self.one), -self.one)
        # grad_change = torch.sign(grad_change_raw)
        return grad_change

    def reset_state(self):
        self.state['grad_prev'].zero_()
        self.state['grad'].zero_()
        self.state['activation_prev'].zero_()
        self.state['activation'].zero_()
        self.state['weight'].zero_()
        self.memory_optimizer.reset_state()

    def init_state(self, grad, weight, cuda):
        grad = grad.view(-1)
        weight = weight.view(-1)
        self.state['grad_prev'] = grad.clone().zero_()
        self.state['grad'] = grad.clone().zero_()
        D = grad.size()[0]
        self.state['weight'] = weight.clone().zero_()
        self.state['activation'] = torch.zeros(D, self.num_local_models)
        self.state['activation_prev'] = torch.zeros(D, self.num_local_models)
        self.state['values_prev'] = torch.zeros(self.value.size())
        self.one = torch.Tensor(1).fill_(1.0)
        if cuda:
            self.sigmasq = self.sigmasq.cuda()
            self.centers = self.centers.cuda()
            self.value = self.value.cuda()
            self.one = self.one.cuda()
            self.state['activation'] = self.state['activation'].cuda()
            self.state['activation_prev'] = self.state[
                'activation_prev'].cuda()
            self.state['values_prev'] = self.state['values_prev'].cuda()
        self.memory_optimizer.init_state(self.state['values_prev'].clone(), cuda)

    def update_state(self, grad, weight):
        grad = grad.view(-1)
        weight = weight.view(-1)
        self.state['grad_prev'].copy_(self.state['grad'])
        self.state['grad'].copy_(grad)
        self.state['activation_prev'].copy_(self.state['activation'])
        self.state['activation'] = self.compute_activation()

        grad_change = self.compute_grad_change()
        activation = self.state['activation_prev']
        grad_change_reshaped = grad_change.view(-1, 1).expand_as(activation)
        weighted_change = grad_change_reshaped *activation
        self.memory_optimizer.update_state(weighted_change.mean(0))

    def update_memory(self):
        grad_change = self.memory_optimizer.get_grad()
        self.state['values_prev'].copy_(self.value)
        lr = self.memory_optimizer.compute_eta()

        # print('grad_change:', grad_change)
        # print('mem lr:', lr)

        # self.debug_mem = {}
        # self.debug_mem['mem_lr'] = lr
        # self.debug_mem['mem_grad_change'] = grad_change.numpy().copy()
        # self.debug_mem['mem_mean_grad_change'] = mean_grad_change.numpy().copy()
        # self.debug_mem['mem_values_prev'] = self.value.numpy().copy()

        self.value = self.value + lr * grad_change
        self.value = torch.max(
            torch.min(self.value, self.one), self.one * 1e-9)
        # self.debug_mem['mem_values'] = self.value.numpy().copy()

    def compute_activation(self):
        """ activation per gradient dimension, per local model

        activation shape: D x #local models
        """
        grad = self.state['grad']
        activation = self.state['activation']

        grad_reshaped = grad.view(-1, 1).expand_as(activation)
        # ipdb.set_trace()
        centers_reshaped = self.centers.view(1, -1).expand_as(activation)
        mahal_dist = (grad_reshaped - centers_reshaped)**2 / self.sigmasq
        return torch.exp(-0.5 * mahal_dist)

    def compute_eta(self):
        activation = self.state['activation']
        activation_total = activation.sum(1)
        eta = (activation *
               self.value.view(1, -1).expand_as(activation)).sum(1)
        eta = eta / activation_total
        return eta
