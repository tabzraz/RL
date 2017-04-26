import numpy as np


class TreeDensity:

    def __init__(self, alphabet_size, context_length, x_shape, y_shape):
        self.alphabet_size = alphabet_size
        self.context_length = context_length
        self.x_shape = x_shape
        self.y_shape = y_shape

        self.count = 1
        self.counts = np.ones(shape=(x_shape, y_shape, self.alphabet_size ** (self.context_length + 1)), dtype=np.int)

        self.to_index = np.array(reversed([alphabet_size ** i for i in range(context_length + 1)]))

    def log_prob(self, contexts, symbols):
        # print(context, symbol)
        # index = 0
        # base = 1
        # for c in reversed(context + [symbol]):
        #     index += c * base
        #     base *= self.alphabet_size
        index = np.dot(self.to_index, contexts)
        print(index)
        return np.log(self.counts[index]) - np.log(self.count)

    def update(self, contexts, symbol):
        index = np.dot(self.to_index, contexts)
        print(index)
        return np.log(self.counts[index]) - np.log(self.count)

        # print(context, symbol)
        # index = 0
        # base = 1
        # for c in reversed(context + [symbol]):
        #     # print(index)
        #     index += c * base
        #     base *= self.alphabet_size
        # log_prob = np.log(self.counts[index]) - np.log(self.count)
        # self.counts[index] += 1
        # self.count += 1
        # return log_prob
