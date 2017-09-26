import Replay.DND_Utils.kdtree as kdtree
from .DND_Utils.lru_cache import LRUCache
import numpy as np
import torch

from torch import Tensor, FloatTensor
from torch.autograd import Variable

# Taken from https://github.com/mjacar/pytorch-nec/blob/master/dnd.py


class DND:
    def __init__(self, kernel, num_neighbors, max_memory, embedding_size):
        self.dictionary = LRUCache(max_memory)
        self.kd_tree = kdtree.create(dimensions=embedding_size)

        self.num_neighbors = num_neighbors
        self.kernel = kernel
        self.max_memory = max_memory

    def is_present(self, key):
        return self.dictionary.get(tuple(key.data.cpu().numpy()[0])) is not None

    def get_value(self, key):
        return self.dictionary.get(tuple(key.data.cpu().numpy()[0]))

    def lookup(self, lookup_key):
        # TODO: Speed up search knn
        keys = [key[0].data for key in self.kd_tree.search_knn(lookup_key, self.num_neighbors)]
        # print(keys)
        output, kernel_sum = Variable(FloatTensor([0])), Variable(FloatTensor([0]))
        # if len(keys) != 0:
            # print(keys)
        # TODO: Speed this up since the kernel takes a significant amount of time
        for key in keys:
            if not (key == lookup_key).data.all():
                # print("Here")
                output += self.kernel(key, lookup_key) * self.dictionary.get(tuple(key.data.cpu().numpy()[0]))
                # print("Key", key.requires_grad, key.volatile)
                # print("Kernel key", self.kernel(key, lookup_key).requires_grad)
                # print("Output in loop", output.requires_grad)
                kernel_sum += self.kernel(key, lookup_key)
                # print(kernel_sum)
        # if len(keys) == 0:
        #     return (lookup_key * 0)[0][0]
        if isinstance(kernel_sum, int):
            return (lookup_key * 0)[0][0]
        # if kernel_sum == 0:
            # print("0 Kernel", kernel_sum)
        # if len(keys) == 0:
            # print("0 keys", len(keys))
        if kernel_sum.data[0] == 0 or len(keys) == 0:
            # print(lookup_key)
            # zeroed = (lookup_key * 0)[0][0]
            # print("Zero Lookup.", output.data, kernel_sum.data, len(keys))
            return (lookup_key * 0)[0][0]
        # print("lookup_key", lookup_key.requires_grad, lookup_key.volatile)
        # print("kernled", self.kernel(keys[0], lookup_key).requires_grad)
        # print("output", output.requires_grad, output.volatile)
        # print("ks", kernel_sum.requires_grad, kernel_sum.volatile)
        # print("Non-Zero Lookup for {}".format(lookup_key))
        output = output / kernel_sum
        # print(output)
        return output

    def upsert(self, key, value):
        if not self.is_present(key):
            self.kd_tree.add(key)

        if self.dictionary.size() == self.max_memory:
            # Expel least recently used key from self.dictionary and self.kd_tree if memory used is at capacity
            deleted_key = self.dictionary.delete_least_recently_used()[0]
            # print("Deleted key:",deleted_key)
            deleted_key = np.array(deleted_key)
            # thing = Variable(torch.from_numpy(deleted_key).float()).unsqueeze(0)
            # thing = Variable(FloatTensor(deleted_key)).unsqueeze(0)
            # print("Thing:",thing)
            self.kd_tree.remove(Variable(FloatTensor(deleted_key)).unsqueeze(0))
            # self.kd_tree.remove(deleted_key)

        self.dictionary.set(tuple(key.data.cpu().numpy()[0]), value)
