
class Interface(object):

    def __init__(self):
        return

    def compute_grad_data(self, grad_data):
        raise NotImplementedError

    def compute_activation(self, grad_data):
        raise NotImplementedError

    def update_memory(self, grad_cur, grad_prev, grad_data, activation):
        raise NotImplementedError

    def compute_eta(self, activation):
        raise NotImplementedError
