# import Replay.DND_Utils.kdtree as kdtree
# from .DND_Utils.lru_cache import LRUCache
import numpy as np
import torch

from torch import Tensor, FloatTensor
from torch.autograd import Variable

# from .DND_Utils.lshash import LSHash
from lru import LRU

# from nearpy import Engine
# from nearpy.hashes import RandomBinaryProjections
# from nearpy.distances import EuclideanDistance
# from nearpy.filters import NearestFilter

# from scipy.spatial.ckdtree import cKDTree as KDTree
from sklearn.neighbors.ball_tree import BallTree as KDTree

# Taken from https://github.com/mjacar/pytorch-nec/blob/master/dnd.py


class DND:
    def __init__(self, kernel, num_neighbors, max_memory, embedding_size):
        # self.dictionary = LRUCache(max_memory)
        # self.kd_tree = kdtree.create(dimensions=embedding_size)
        # rnd_projection = RandomBinaryProjections("RBP", 8)
        # distance = EuclideanDistance()
        # nearest = NearestFilter(num_neighbors)
        # self.nearpy = Engine(dim=embedding_size, lshashes=[rnd_projection], distance=distance, vector_filters=[nearest], fetch_vector_filters=[])

        self.kd_tree = None
        # self.data = []

        # self.lshash = LSHash(hash_size=embedding_size, input_dim=embedding_size, num_hashtables=10)
        self.lru = LRU(size=max_memory)

        self.num_neighbors = num_neighbors
        self.kernel = kernel
        self.max_memory = max_memory
        self.embedding_size = embedding_size
        # self.keys_added = []

    def is_present(self, key):
        return tuple(key) in self.lru  # self.lru.has_key(tuple(key))
        # return self.dictionary.get(tuple(key)) is not None
        # return self.dictionary.get(tuple(key.data.cpu().numpy()[0])) is not None

    def get_value(self, key):
        return self.lru[tuple(key)]
        # return self.dictionary.get(tuple(key))
        # return self.dictionary.get(tuple(key.data.cpu().numpy()[0]))

    def lookup(self, lookup_key):
        # TODO: Speed up search knn
        # keys = [key[0].data for key in self.kd_tree.search_knn(lookup_key, self.num_neighbors)]
        lookup_key_numpy = lookup_key.data[0].numpy()
        # lookup_key_tuple = tuple(lookup_key_numpy)
        # print(lookup_key)

        # keys = [key[0] for key in self.lshash.query_no_data(lookup_key_numpy, num_results=self.num_neighbors)]
        # keys = [key[1] for key in self.nearpy.neighbours(lookup_key_numpy)]
        if self.kd_tree is not None:
            # print(len(self.lru.keys()), lookup_key_numpy)
            # things_distances, things_index = self.kd_tree.query(lookup_key_numpy, k=self.num_neighbors, eps=1.0)
            things_index = self.kd_tree.query([lookup_key_numpy], k=min(self.num_neighbors, len(self.kd_tree.data)), return_distance=False, sort_results=False)
            # print(things_index)
            keys = [self.lru.keys()[ii[0]] for ii in things_index]
            # print(keys)
        else:
            keys = []

        # print(keys)
        # print(keys)
        # output, kernel_sum = Variable(FloatTensor([0])), Variable(FloatTensor([0]))
        output, kernel_sum = 0, 0
        # if len(keys) != 0:
            # print(keys)
        # TODO: Speed this up since the kernel takes a significant amount of time
        for key in keys:
            # print("Key:",key, lookup_key)
            # if not np.allclose(key, lookup_key_numpy): #(key == lookup_key).data.all():
            if not np.all(key == lookup_key_numpy):
                # print("Here")
                # gg = Variable(FloatTensor(np.array(key)))
                # print(key)
                # gg = Variable(FloatTensor(key))
                gg = Variable(torch.from_numpy(np.array(key)))
                # print(tuple(key))
                # hh = lookup_key[0] - gg
                # print("Key:", gg, "Lookup key", lookup_key[0])
                # print(lookup_key[0] + gg)
                kernel_val = self.kernel(gg, lookup_key[0])
                # print("key:", self.lru.get(tuple(key)))
                # if not self.lru.has_key(tuple(key)):
                    # print(keys)
                    # print(tuple(key))
                    # print(key in self.keys_added)
                    # print(len(self.lru))
                # if tuple(key) not in self.lru:
                    # print("NOT IN:", tuple(key))
                    # print(len(keys))
                output += kernel_val * self.lru.get(tuple(key))
                # output += kernel_val * self.dictionary.get(tuple(key))
                # print("Key", key.requires_grad, key.volatile)
                # print("Kernel key", self.kernel(key, lookup_key).requires_grad)
                # print("Output in loop", output.requires_grad)
                kernel_sum += kernel_val #self.kernel(key, lookup_key)
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
        # key = key.data[0].numpy()
        # print(key)
        # self.keys_added.append(key)
        # if not self.lru.has_key(tuple(key)):# self.is_present(key):
            # self.kd_tree.add(key)
            # print("Key going in", key)
        # self.lshash.index(input_point=key)
        # self.nearpy.store_vector(key, data=key)

        # print("Adding", tuple(key), key)
        # neighbours = self.nearpy.neighbours(key)
        # print(neighbours)

        self.lru[tuple(key)] = value
        # self.kd_tree = KDTree(data=self.lru.keys(), compact_nodes=False, copy_data=False, balanced_tree=False)
        self.kd_tree = KDTree(self.lru.keys())

        return
        if len(self.lru) == self.max_memory:
            # Expel least recently used key from self.dictionary and self.kd_tree if memory used is at capacity
            # deleted_key = self.dictionary.delete_least_recently_used()[0]
            # deleted_key = self.lru.peek_last_item()[0]
            # print("Deleted key:",deleted_key)
            # deleted_key = np.array(deleted_key)
            # thing = Variable(torch.from_numpy(deleted_key).float()).unsqueeze(0)
            # thing = Variable(FloatTensor(deleted_key)).unsqueeze(0)
            # print("Thing:",thing)
            # print(self.dictionary.cache.keys())
            key_to_delete = self.lru.peek_last_item()
            self.lru[tuple(key)] = value
            # self.kd_tree.remove(Variable(FloatTensor(deleted_key)).unsqueeze(0))
            # self.kd_tree.remove(deleted_key)

            # Remake the LSHASH with the deleted key
            # print("remaking")

            # self.lshash = LSHash(hash_size=self.embedding_size, input_dim=self.embedding_size)
            # for k in self.lru.keys():
            #     self.lshash.index(np.array(k))

            # print("Deleting", np.array(key_to_delete[0]))
            # self.nearpy.delete_vector(key_to_delete[0])
            # self.nearpy.clean_all_buckets()
            # for k in self.lru.keys():
                # self.nearpy.store_vector(np.array(k))


            # Checking that the lru keys are the same as the keys in the lshash
            # for key in self.lru.keys():
            #     keys_close = [key[0] for key in self.lshash.query(key, num_results=5)]
            #     # print(keys_close)
            #     for kk in keys_close:
            #         if kk not in self.lru:
            #             print("\n\nProblems! Key in LSHASH not in LRU\n\n")

            # Check length of all lru keys
            # all_lru_keys = self.lshash.query(key)
            # print("\n", len(all_lru_keys), "\n")
        else:
            self.lru[tuple(key)] = value

        self.kdtree = KDTree(self.data)
        # self.dictionary.set(tuple(key), value)
