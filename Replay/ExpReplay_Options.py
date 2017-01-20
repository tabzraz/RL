import numpy as np


class ExperienceReplay_Options:

    def __init__(self, N):
        self.N = N
        self.Exps = []

    def Add_Exp(self, state_now, action, reward, state_after, steps, terminal):
        if len(self.Exps) >= self.N:
            self.Exps.pop(0)
        # Make copies just in case
        self.Exps.append((np.copy(state_now), action, reward, np.copy(state_after), steps, terminal))

    def Sample(self, size):
        assert(size <= self.N)
        if len(self.Exps) < size:
            return self.Exps
        indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        return [self.Exps[i] for i in indices]
