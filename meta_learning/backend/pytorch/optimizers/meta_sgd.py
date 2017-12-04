
import os

#from tf_utils import utils as _tf_utils
import torch
from torch.optim.optimizer import Optimizer, required

from meta_learning.backend.pytorch.optimizers import memory_static_1d


class MetaSgd(Optimizer):
    """Implements Sgd with meta learning algorithm. named_params
        (iterable): iterable of parameters to optimize or dicts
        defining parameter groups

        lr (float, optional): learning rate (default: 1e-3)

        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))

        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self,
                 model,
                 memory_type=required,
                 memory_optimizer_init_fn=required,
                 lr=required,
                 clip_grad=required,
                 meta_params=required,
                 logdir=required,
                 debug=False,
                 weight_decay=0):

        defaults = dict(lr=lr,
                        weight_decay=weight_decay)
        self._param_names = {}
        self._use_memory = {}
        params = self._init_params(model.named_parameters(), meta_params)
        super(MetaSgd, self).__init__(params, defaults)
        self._clip_grad = clip_grad
        print('clip_grad: ', clip_grad)
        self._meta_params = meta_params
        self._memory_type = memory_type
        self._is_debug = debug
        self._logdir = logdir
        self._step = 0
        self._cuda = False
        self._memory_optimizer_init_fn = memory_optimizer_init_fn

    def cuda(self):
        self._cuda = True
        if self._is_debug:
            print "cuda set to true, setting debug mode to false"
            self._is_debug = False

    def _init_params(self, named_params, meta_params):
        self._param_names = {}
        self._use_memory = {}
        params = []
        for (name, param) in named_params:
            param_id = self._get_param_id(param)
            params.append(param)
            self._param_names[param_id] = name
            if isinstance(meta_params['use_memory'], list):
                if name in meta_params['use_memory']:
                    self._use_memory[param_id] = True
                else:
                    self._use_memory[param_id] = False
            else:
                self._use_memory[param_id] = meta_params['use_memory']
        return params

    def _get_param_id(self, param):
        return id(param)

    def _init_optimizer_state(self, state, use_memory, lr, grad, weight):
        state['step'] = 0
        if use_memory:
            state['memory'] = self._memory_type(self._meta_params,
                                                lr,
                                                self._memory_optimizer_init_fn,
                                                self._is_debug)
            state['memory'].init_state(grad,
                                       weight,
                                       cuda=self._cuda)

        return state

    def _create_debug_info(self,
                           memory,
                           param_grad,
                           param_val,
                           scaled_grad,
                           lr):
        debug_data = {}
        debug_data['param_val'] = param_val.numpy().copy()
        debug_data['param_grad'] = param_grad.numpy().copy()
        debug_data['lr'] = lr.numpy().copy()
        debug_data['scaled_grad_cur'] = scaled_grad.numpy().copy()

        if memory is not None:
            mem_state = memory.state
            debug_data['values_prev'] = mem_state['values_prev'].numpy().copy()
            debug_data['grad_cur'] = mem_state['grad'].numpy().copy()
            debug_data['grad_prev'] = mem_state['grad_prev'].numpy().copy()
            debug_data['values'] = memory.value.numpy().copy()
            debug_data['activation'] = mem_state['activation'].numpy().copy()
            #debug_data['grad_change'] = mem_state['grad_change'].numpy().copy()
            debug_data['centers'] = memory.centers.numpy().copy()
            debug_data['sigmasq'] = memory.sigmasq
            # debug_data['debug_mem'] = memory.debug_mem.copy()
        # for key, value in sorted(debug_data.items()):
            # if key != 'lr':
            #     continue
            # print('{}: {}'.format(key, value))
        return debug_data

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self._is_debug:
            debug_data = {}

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                param_id = self._get_param_id(param)
                param_name = self._param_names[param_id]
                param.grad.data.clamp_(min=-self._clip_grad,
                                       max=self._clip_grad)
                param_grad = param.grad.data

                param_val = param.data

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    self._init_optimizer_state(state,
                                               self._use_memory[param_id],
                                               group['lr'],
                                               param_grad,
                                               param_val)

                # if self._use_memory[param_id]:
                if self._use_memory[param_id]:
                    memory = state['memory']
                    memory.update_state(param_grad, param_val)
                    memory.update_memory()
                    lr = memory.compute_eta()
                    # debug print
                    # print('lr   ', lr.numpy())
                    # print('grad ', param_grad.numpy())
                    scaled_grad = lr.view(param_grad.size()) * param_grad
                else:
                    lr = group['lr']
                    scaled_grad = lr * param_grad

                if self._is_debug and not self._cuda:
                    debug_data[param_name] = self._create_debug_info(
                        memory, param_grad, param_val, scaled_grad, lr)

                # print('grad_apply ', scaled_grad.numpy())

                # print('lr: ', lr.numpy())
                # print('grad: ', param_grad.numpy())
                # print('scaled_grad:', scaled_grad.numpy())
                param.data.add_(-1.0, scaled_grad)
                state['step'] += 1

        #if self._is_debug and not self._cuda:
        #    file_path = os.path.join(self._logdir,
        #                             "debug_iter_{}.pkl".format(self._step))
        #    _tf_utils.pkl_save(file_path, debug_data)

        self._step += 1

        return loss

    def save(self, optimizer_path):
        torch.save(self.state_dict(), optimizer_path)
        return

    def load(self, optimizer_path):
        self.load_state_dict(torch.load(optimizer_path))
        self._step = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['memory'].reset_state()

        print "optimizer from path: {} loaded".format(optimizer_path)
        print "reset step to zero"
        print "state of optimizer"
        print self.state_dict()
        return


class MetaSgd1d(MetaSgd):
    """Implements Sgd with meta learning algorithm. named_params
        (iterable): iterable of parameters to optimize or dicts
        defining parameter groups

        lr (float, optional): learning rate (default: 1e-3)

        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))

        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self,
                 model,
                 memory_optimizer_init_fn,
                 lr=required,
                 clip_grad=required,
                 meta_params=required,
                 logdir=required,
                 debug=False,
                 weight_decay=0):
        super(MetaSgd1d, self).__init__(
            model=model,
            memory_type=memory_static_1d.MemoryStatic1d,
            memory_optimizer_init_fn=memory_optimizer_init_fn,
            lr=lr,
            clip_grad=clip_grad,
            meta_params=meta_params,
            logdir=logdir,
            debug=debug,
            weight_decay=weight_decay)

