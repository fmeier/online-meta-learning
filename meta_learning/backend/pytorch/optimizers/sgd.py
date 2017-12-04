from torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    """Implements SGD with coordinate-wise gradient clipping.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=required, clip_grad=required,
                 momentum=0, dampening=0, weight_decay=0,
                 nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires" +
                             " a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)
        self.clip_grad = clip_grad
        print('clip_grad: ', clip_grad)

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
                # print grad

                lr = group['lr']
                scaled_grad = lr * grad
                # scaled_grad.clamp_(min = -self.clip_grad,
                #                    max=self.clip_grad)

                p.data.add_(-1.0, scaled_grad)

        return loss
