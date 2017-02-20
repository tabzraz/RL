import numpy as np
from collections import namedtuple
from math import pow
from .Binary_Heap import BinaryHeap

Experience = namedtuple("Experience", ["State", "Action", "Reward", "State_Next", "Steps", "Terminal"])


class ExperienceReplay_Options:

    def __init__(self, N, alpha=0.7):
        self.N = N
        self.alpha = alpha
        self.Exps = [None for _ in range(N)]
        self.storing_index = 0
        self.experiences_stored = 0

        self.priorities = BinaryHeap(N)

        # self.priorities = [(i, -100) for i in range(N)]
        # self.exp_i_to_p_i = {}
        # for i in range(N):
        #     self.exp_i_to_p_i[i] = i
        # self.p_i_to_exp_i = {}
        # for i in range(N):
        #     self.p_i_to_exp_i[i] = i

    def Add_Exp(self, state_now, action, reward, state_after, steps, terminal):
        if self.storing_index >= self.N:
            self.storing_index = 0
        # Make copies just in case
        self.Exps[self.storing_index] = Experience(State=np.copy(state_now), Action=action, Reward=reward, State_Next=np.copy(state_after), Steps=steps, Terminal=terminal)

        priority = self.priorities.get_max_priority()
        self.priorities.update(priority, self.storing_index)

        # priority_index = self.exp_i_to_p_i[self.storing_index]
        # priority_index = list(map(lambda x: x[0], self.priorities)).index(self.storing_index)
        # self.priorities[priority_index] = (self.storing_index, 100)

        # self.p_i_to_exp_i[priority_index] = self.storing_index

        # self.priorities.sort(key=lambda x: x[1])

        self.storing_index += 1
        self.experiences_stored = max(self.experiences_stored, self.storing_index)

        # Balance the tree every now and again
        if self.storing_index == self.N:
            self.priorities.balance_tree()

    def sampling_distribution(self):
        distrib = np.array([pow((1 / (i + 1)), self.alpha) for i in range(self.experiences_stored)])
        prob_sum = np.sum(distrib)
        return distrib / prob_sum

    def get_indices(self, size, prioritized=True):
        indices = None
        if prioritized:
            distribution = self.sampling_distribution()
            sampled_indices = np.random.choice([i + 1 for i in range(self.experiences_stored)], p=distribution, size=size)
            # indices = [self.priorities[self.N - 1 - rank_index][0] for rank_index in sampled_indices]
            # print(sampled_indices)
            indices = self.priorities.priority_to_experience(sampled_indices)
        else:
            indices = np.random.randint(low=0, high=self.experiences_stored, size=size)
        return indices

    def Update_Indices(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            # p_index = list(map(lambda x: x[0], self.priorities)).index(index)
            # p_index = self.exp_i_to_p_i[index]
            # self.priorities[p_index] = (index, priority)
            self.priorities.update(priority, index)
        # self.priorities.sort(key=lambda x: x[1])

    def Sample(self, size):
        indices = self.get_indices(size)
        return [self.Exps[i] for i in indices]

    # Sample an n-step target
    # Intended for use with step sizes of 1 for now
    def Sample_N(self, size, N, gamma):
        # indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        indices = self.get_indices(size)
        batch_to_return = []
        for index in indices:
            exps_to_use = self.Exps[index: index + N]
            # Check for terminal states
            index_up_to = N
            for i, exp in enumerate(exps_to_use):
                if exp[5]:
                    index_up_to = i + 1
                    break
            exps_to_use = exps_to_use[:index_up_to]

            state_now = exps_to_use[0][0]
            action_now = exps_to_use[0][1]
            state_then = exps_to_use[-1][3]
            terminate = exps_to_use[-1][5]
            steps = len(exps_to_use)
            rewards_to_use = list(map(lambda x: x[2], exps_to_use))
            accum_reward = 0
            for ri in reversed(rewards_to_use):
                accum_reward = ri + gamma * accum_reward
            new_exp = (state_now, action_now, accum_reward, state_then, steps, terminate)
            batch_to_return.append(new_exp)
        return indices, batch_to_return
