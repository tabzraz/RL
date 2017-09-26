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
        output, kernel_sum = 0, 0
        # TODO: Speed this up since the kernel takes a significant amount of time
        for key in keys:
            if not (key == lookup_key).data.all():
                output += self.kernel(key, lookup_key) * self.dictionary.get(tuple(key.data.cpu().numpy()[0]))
                kernel_sum += self.kernel(key, lookup_key)
        if kernel_sum == 0 or len(keys) == 0:
            # print(lookup_key)
            # zeroed = (lookup_key * 0)[0][0]
            # print(zeroed)
            return (lookup_key * 0)[0][0]
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
