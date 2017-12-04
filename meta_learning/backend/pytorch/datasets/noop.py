
import numpy as np
import torch

from meta_learning.backend.pytorch.datasets import interface as _interface


class NoopData(_interface.Interface):
    """Just a fake dataset class if there is not input required.
    """

    @classmethod
    def create_from_params(cls, params, *args, **kwargs):
        return cls(params.batch_size,
                   params.shard_name)

    def __init__(self,
                 batch_size,
                 shard_name):
        super(NoopData, self).__init__(batch_size=batch_size,
                                       shard_name=shard_name)

    @property
    def num_examples(self):
        """Return the number of examples in the dataset."""
        return 1

    def get_next_batch(self):
        return {'x': None, 'y': torch.Tensor(np.zeros(1))}
