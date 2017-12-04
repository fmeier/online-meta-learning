
import numpy as np
import torch
# import torchvision.datasets as datasets
from torchvision import datasets, transforms

from meta_learning.backend.pytorch.datasets import interface as _interface

class Mnist(_interface.Interface):
    """Just the standard mnist example.
    """

    def __init__(self, batch_size, shard_name, shuffle, data_dir):
        super(Mnist, self).__init__(batch_size=batch_size,
                                    shard_name=shard_name)
        self._data_dir = data_dir

        if shard_name == _interface.SHARD_NAME_TRAIN:
            self.dataset = torch.utils.data.DataLoader(
                datasets.MNIST(data_dir, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=batch_size, shuffle=shuffle)

        if shard_name == _interface.SHARD_NAME_TEST:
            self.dataset = torch.utils.data.DataLoader(
                datasets.MNIST(data_dir, train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=batch_size, shuffle=shuffle)

        self._shuffle = shuffle
        self._batch_size = batch_size

    @property
    def num_examples(self):
        if self._shard_name == _interface.SHARD_NAME_TRAIN:
            return len(self.dataset.data)
        if self._shard_name == _interface.SHARD_NAME_TEST:
            return len(self.dataset.data)

    def get_next_batch(self):
        if self._shard_name == _interface.SHARD_NAME_TRAIN:
            (images, labels) = self.dataset.__iter__().next()
            return {'x': images, 'y': labels}
        if self._shard_name == _interface.SHARD_NAME_TEST:
            (images, labels) = self.dataset.__iter__().next()
            return {'x': images, 'y': labels}


class MnistClassPair(Mnist):

    def __init__(self, batch_size,
                 pos_class,
                 neg_class,
                 shard_name,
                 seed,
                 data_dir):
        super(MnistClassPair, self).__init__(batch_size=batch_size,
                                             shard_name=shard_name,
                                             shuffle=False,
                                             data_dir=data_dir)

        self._pos_class = pos_class
        self._neg_class = neg_class

        if self._shard_name == _interface.SHARD_NAME_TRAIN:
            self._pos_idx = torch.nonzero(self.dataset.dataset.train_labels ==
                                          pos_class)
            self._neg_idx = torch.nonzero(self.dataset.dataset.train_labels ==
                                          neg_class)


        if shard_name == _interface.SHARD_NAME_TEST:
            self._pos_idx = torch.nonzero(self.dataset.dataset.test_labels ==
                                          pos_class)
            self._neg_idx = torch.nonzero(self.dataset.dataset.test_labels ==
                                          neg_class)

        idx = torch.cat([self._pos_idx, self._neg_idx], dim=0)
        torch.manual_seed(seed)

        rand_order = torch.randperm(len(idx))
        self._idx = idx[rand_order]
        self._step = 0

        if self._batch_size == -1:
            self._batch_size = len(self._idx)
            print "({}) Using all data per batch".format(shard_name)

    def get_next_batch(self):
        
        # idx = torch.randperm(len(self._idx))[:self._batch_size]
        if (self._step+1)*self._batch_size > len(self._idx):
            # print "({}) cycled through data, restart".format(self._shard_name)
            self._step = 0

        start = self._step*self._batch_size
        end = (self._step+1)*self._batch_size
        binary_idx = self._idx[start:end, 0]
        # print binary_idx
        # binary_idx = self._idx.index_select(dim=0, index=idx)#[:, 0]
        
        if self._shard_name == _interface.SHARD_NAME_TRAIN:
            x = self.dataset.dataset.train_data[binary_idx, :].float()
            y = self.dataset.dataset.train_labels[binary_idx]
        if self._shard_name == _interface.SHARD_NAME_TEST:
            x = self.dataset.dataset.test_data[binary_idx, :].float()
            y = self.dataset.dataset.test_labels[binary_idx]

        # transform y to 0 and 1, pos_class = 1, neg_class = 0
        y[y==self._pos_class] = 1
        y[y==self._neg_class] = 0 

        x_shape = x.shape
        self._step += 1
        return {'x': x.view(x_shape[0], 1, x_shape[1], x_shape[2]), 'y': y} 

    def reset(self):
        self._step = 0
