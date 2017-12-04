import torch
from torch.optim import Optimizer
from meta_learning.backend.pytorch.optimizer.memory_static import MemoryStatic


class MetaSGD(Optimizer):
    """Implements SGD with meta learning algorithm.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, use_memory=True, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(MetaSGD, self).__init__(params, defaults)
        self.clip_grad = 2
        self.use_memory = use_memory

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data.clamp_(min=-self.clip_grad,
                                   max=self.clip_grad)
                grad = p.grad.data

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.FloatTensor(grad.size())
                    if self.use_memory:
                        state['memory'] = MemoryStatic(self.clip_grad,
                                                       20,
                                                       0.1,
                                                       0.01,
                                                       group['lr'],
                                                       True)

                if self.use_memory:
                    memory = state['memory']
                    if state['step'] > 0:
                        prev_grad = state['prev_grad']
                        memory.update_memory(prev_grad.view(-1), grad.view(-1))

                    lr = memory.compute_eta(grad.view(-1))
                    scaled_grad = lr.view(grad.size()) * grad
                    state['prev_grad'].copy_(grad)
                else:
                    lr = group['lr']
                    scaled_grad = lr * grad

                p.data.add_(-1.0, scaled_grad)
                state['step'] += 1

        return loss


class MetaAdam(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, use_memory=True, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(MetaSGD, self).__init__(params, defaults)
        self.clip_grad = 2
        self.use_memory = use_memory

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data.clamp_(min=-self.clip_grad,
                                   max=self.clip_grad)
                grad = p.grad.data

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.FloatTensor(grad.size())
                    if self.use_memory:
                        state['memory'] = MemoryStatic(self.clip_grad,
                                                       20,
                                                       0.1,
                                                       0.01,
                                                       group['lr'],
                                                       True)

                if self.use_memory:
                    memory = state['memory']
                    if state['step'] > 0:
                        prev_grad = state['prev_grad']
                        memory.update_memory(prev_grad.view(-1), grad.view(-1))

                    lr = memory.compute_eta(grad.view(-1))
                    scaled_grad = lr.view(grad.size()) * grad
                    state['prev_grad'].copy_(grad)
                else:
                    lr = group['lr']
                    scaled_grad = lr * grad

                p.data.add_(-1.0, scaled_grad)
                state['step'] += 1

        return loss
