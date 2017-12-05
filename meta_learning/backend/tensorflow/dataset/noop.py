
from meta_learning.backend.tensorflow.dataset import interface as _interface


class Noop(_interface.Interface):
    """Just a fake dataset class if there is not input required.
    """

    @classmethod
    def create_from_params(cls, params, *args, **kwargs):
        return cls(params.num_epochs,
                   params.batch_size,
                   params.shard_name)

    def __init__(self,
                 num_epochs,
                 batch_size,
                 shard_name):
        super(Noop, self).__init__(batch_size=batch_size,
                                   num_epochs=num_epochs,
                                   shard_name=shard_name)

    @property
    def num_examples(self):
        """Return the number of examples in the dataset."""
        return 1

    def create_input_fn(self):
        def input_fn():
            return {}, {}
        return input_fn
