import numpy as np
import collections


class ExperienceReplay_Options_Pseudo:

    Experience = collections.namedtuple("Experience", "state action reward state_next steps terminal pseudo_reward pseudo_reward_t")

    def __init__(self, N, pseudo_limit, exp_model):
        self.N = N
        self.Exps = []
        self.pseudo_limit = pseudo_limit
        self.exp_model = exp_model
        self.T = 0

    def Clear(self):
        self.Exps = []

    def Add_Exp(self, state_now, action, reward, state_after, steps, terminal, pseudo_reward=0):
        if len(self.Exps) >= self.N:
            self.Exps.pop(0)
        # Make copies just in case
        new_exp = self.Experience(state=np.copy(state_now), action=action, reward=reward, state_next=np.copy(state_after), steps=steps, terminal=terminal, pseudo_reward=pseudo_reward, pseudo_reward_t=self.T)
        self.Exps.append(new_exp)
        # self.Exps.append((np.copy(state_now), action, reward, np.copy(state_after), steps, terminal, psuedo_reward, self.T))

    def Recompute_Pseudo_Counts(self, indices):
        if self.exp_model is None:
            # print("No exp model")
            return
        for i in indices:
            exp = self.Exps[i]
            if self.T - exp.pseudo_reward_t > self.pseudo_limit:
                # Recompute it
                new_bonus, _ = self.exp_model.bonus(exp.state, dont_remember=True)
                new_exp = self.Experience(state=exp.state, action=exp.action, reward=exp.reward, state_next=exp.state_next, steps=exp.steps, terminal=exp.terminal, pseudo_reward=new_bonus, pseudo_reward_t=self.T)
                self.Exps[i] = new_exp

    def Sample(self, size):
        assert(size <= self.N)
        if len(self.Exps) < size:
            return self.Exps
        indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        self.Recompute_Pseudo_Counts(indices)
        # sample = [self.Exps[i] for i in indices]
        return [self.Exps[i] for i in indices]

    def Sample_Sequence(self, size, seq_length):
        indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for index in indices:
            exps_to_use = self.Exps[index: index + seq_length]
            # Check for terminal states
            index_up_to = seq_length
            for i, exp in enumerate(exps_to_use):
                if exp[5]:
                    index_up_to = i + 1
                    break
            exps_to_use = exps_to_use[:index_up_to]

            states.append([x[0] for x in exps_to_use])
            actions.append([x[1] for x in exps_to_use])
            rewards.append([x[2] for x in exps_to_use])
            next_states.append([x[3] for x in exps_to_use])
            terminals.append([x[5] for x in exps_to_use])

        return states, actions, rewards, next_states, terminals

    # Sample an n-step target
    # Intended for use with step sizes of 1 for now
    def Sample_N(self, size, N, gamma):
        assert(size <= self.N)
        self.T += 1
        indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        self.Recompute_Pseudo_Counts(indices)
        batch_to_return = []
        for index in indices:
            exps_to_use = self.Exps[index: index + N]
            # Check for terminal states
            index_up_to = N
            for i, exp in enumerate(exps_to_use):
                if exp.terminal:
                    index_up_to = i + 1
                    break
            exps_to_use = exps_to_use[:index_up_to]

            state_now = exps_to_use[0].state
            action_now = exps_to_use[0].action
            state_then = exps_to_use[-1].state_next
            terminate = exps_to_use[-1].terminal
            steps = len(exps_to_use)
            rewards_to_use = list(map(lambda x: x.reward + x.pseudo_reward, exps_to_use))
            accum_reward = 0
            for ri in reversed(rewards_to_use):
                accum_reward = ri + gamma * accum_reward
            new_exp = (state_now, action_now, accum_reward, state_then, steps, terminate)
            batch_to_return.append(new_exp)
        # [] for indices to match the prioritized replay
        return batch_to_return
