import numpy as np


class ExperienceReplay_Options:

    def __init__(self, N):
        self.N = N
        self.Exps = []

    def Clear(self):
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

    # Sample an n-step target
    # Intended for use with step sizes of 1 for now
    def Sample_N(self, size, N, gamma):
        assert(size <= self.N)
        indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
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
        # [] for indices to match the prioritized replay
        return [], batch_to_return
