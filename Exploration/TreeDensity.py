import numpy as np


class TreeDensity:

    def __init__(self, alphabet_size, context_length, x_shape, y_shape):
        self.alphabet_size = alphabet_size
        self.context_length = context_length
        self.x_shape = x_shape
        self.y_shape = y_shape

        self.counts = np.ones(shape=(x_shape, y_shape, self.alphabet_size ** (self.context_length + 1)), dtype=np.uint32)
        print("Memory for TreeDensity Model is {} GB".format(self.counts.nbytes / (1024 ** 3)))
        self.count = np.sum(self.counts)
        # print(self.counts.shape)

        self.to_index = np.array(list(reversed([alphabet_size ** i for i in range(context_length + 1)])))
        # print(self.to_index)

    def new_old(self, contexts, keep=True):
        index = np.dot(contexts, self.to_index)
        index = index.astype(np.int)
        x, y = np.ogrid[:self.counts.shape[0], :self.counts.shape[1]]
        prev_log_prob = np.log(self.counts[x, y, index]) - np.log(self.count)
        new_prob = np.log(self.counts[x, y, index] + 1) - np.log(self.count + 1)
        if keep:
            self.count += 1
            self.counts[x, y, index] += 1
        return new_prob - prev_log_prob, prev_log_prob

    def log_prob(self, contexts):
        # print(context, symbol)
        # index = 0
        # base = 1
        # for c in reversed(context + [symbol]):
        #     index += c * base
        #     base *= self.alphabet_size
        index = np.dot(contexts, self.to_index)
        index = index.astype(np.int)
        x, y = np.ogrid[:self.counts.shape[0], :self.counts.shape[1]]
        # print(self.counts[x, y, index])
        # print(self.counts.choose(index))
        # index = 
        # print(np.log(self.counts[index]).shape)
        return np.log(self.counts[x, y, index]) - np.log(self.count)

    def update(self, contexts):
        # print(context, symbol)
        # index = 0
        # base = 1
        # for c in reversed(context + [symbol]):
        #     index += c * base
        #     base *= self.alphabet_size
        index = np.dot(contexts, self.to_index)
        index = index.astype(np.int)
        x, y = np.ogrid[:self.counts.shape[0], :self.counts.shape[1]]
        # print(self.counts[x, y, index])
        # print(self.counts.choose(index))
        # index = 
        # print(np.log(self.counts[index]).shape)
        log_probs = np.log(self.counts[x, y, index]) - np.log(self.count)
        self.counts[x, y, index] += 1
        self.count += 1
        return log_probs

    # def update(self, contexts):
    #     index = np.dot(contexts, self.to_index)
    #     print(index)
    #     return np.log(self.counts[index]) - np.log(self.count)

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
