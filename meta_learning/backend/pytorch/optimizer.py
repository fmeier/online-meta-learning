
from meta_learning.backend.pytorch.optimizers.meta_sgd import MetaSgd1d as MetaSgd
from meta_learning.backend.pytorch.optimizers.sgd import SGD
from meta_learning.backend.pytorch.optimizers.adam import Adam
from meta_learning.backend.pytorch.optimizers import memory_optimizer


def memory_optimizer_sgd_init(lr):
    def fn():
        return memory_optimizer.Sgd(lr=lr)
    return fn


def memory_optimizer_adam_init(lr, eps=1e-8, beta1=0.9, beta2=0.999):
    def fn():
        return memory_optimizer.Adam(lr=lr,
                                     eps=eps,
                                     beta1=beta1,
                                     beta2=beta2)
    return fn
