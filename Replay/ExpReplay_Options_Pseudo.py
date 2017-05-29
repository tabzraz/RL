import numpy as np
import collections
from .Binary_Heap import BinaryHeap


class ExperienceReplay_Options_Pseudo:

    Experience = collections.namedtuple("Experience", "state action reward state_next steps terminal pseudo_reward pseudo_reward_t trajectory_end")

    def __init__(self, N, pseudo_limit, exp_model, priority=False):
        self.N = N
        self.Exps = [None for _ in range(N)]
        self.pseudo_limit = pseudo_limit
        self.exp_model = exp_model
        self.T = 0
        self.storing_index = 0
        self.experiences_stored = 0

        self.priority = priority
        if self.priority:
            self.alpha = 0.7
            self.priorities = BinaryHeap(N)
            self.distrib = np.array([pow((1 / (i + 1)), self.alpha) for i in range(self.N)])
            self.full_distrib = self.distrib / np.sum(self.distrib)

        self.special_start = 0
        self.special_end = 0
        self.trajectory_index = 0

    def Clear(self):
        self.Exps = [None for _ in range(self.N)]
        self.storing_index = 0
        self.experiences_stored = 0

    def Add_Exp(self, state_now, action, reward, state_after, steps, terminal, pseudo_reward=0):
        if self.storing_index >= self.N:
            self.storing_index = 0
        # if len(self.Exps) >= self.N:
            # self.Exps.pop(0)
        # Make copies just in case
        new_exp = self.Experience(state=np.copy(state_now), action=action, reward=reward, state_next=np.copy(state_after), steps=steps, terminal=terminal, pseudo_reward=pseudo_reward, pseudo_reward_t=self.T, trajectory_end=terminal)
        self.Exps[self.storing_index] = new_exp

        if self.priority:
            p = self.priorities.get_max_priority()
            self.priorities.update(p, self.storing_index)
            if self.storing_index % 1000 == 0:
                self.priorities.balance_tree()

        self.storing_index += 1
        self.experiences_stored = max(self.experiences_stored, self.storing_index)

    def end_of_trajectory(self):
        exp = self.Exps[self.storing_index - 1]
        new_exp = list(exp)
        new_exp[-1] = True  # End of trajectory
        new_exp_tuple = self.Experience(*new_exp)
        self.Exps[self.storing_index - 1] = new_exp_tuple

    def get_sampling_distribution(self):
        if self.experiences_stored < self.N:
            partial = self.distrib[:self.experiences_stored]
            return partial / np.sum(partial)
        else:
            return self.full_distrib

    def get_indices(self, size):
        if self.priority:
            distribution = self.get_sampling_distribution()
            sampled_indices = np.random.choice(np.arange(1, self.experiences_stored + 1), p=distribution, size=size)
            indices = self.priorities.priority_to_experience(sampled_indices)
        else:
            indices = np.random.randint(low=0, high=self.experiences_stored, size=size)
        return indices

    def Recompute_Pseudo_Counts(self, indices):
        if self.exp_model is None or self.pseudo_limit >= self.N:
            # print("No exp model")
            return
        for i in indices:
            exp = self.Exps[i]
            if self.T - exp.pseudo_reward_t > self.pseudo_limit:
                # Recompute it
                new_bonus, _ = self.exp_model.bonus(exp.state, dont_remember=True)
                new_exp = self.Experience(state=exp.state, action=exp.action, reward=exp.reward, state_next=exp.state_next, steps=exp.steps, terminal=exp.terminal, pseudo_reward=new_bonus, pseudo_reward_t=self.T)
                self.Exps[i] = new_exp

    def Update_Indices(self, indices, ps):
        if self.priority:
            # print(ps)
            for index, priority in zip(indices, ps):
                # print(index, priority)
                self.priorities.update(priority[0] + 0.00001, index)

    def Sample(self, size):
        # assert(size <= self.N)
        # if len(self.Exps) < size:
            # return self.Exps
        # indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        self.T += 1
        indices = self.get_indices(size)
        self.Recompute_Pseudo_Counts(indices)
        # sample = [self.Exps[i] for i in indices]
        return [self.Exps[i] for i in indices], indices

    def Sample_Sequence(self, size, seq_length):
        # indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        self.T += 1
        indices = self.get_indices(size)
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

    def special_trajectory(self):
        # Special trajectory is [special_start, special_end]
        self.special_end = (self.storing_index - 1) % self.experiences_stored
        i = (self.special_end - 1) % self.experiences_stored
        exp = self.Exps[i]
        while not (exp.terminal or exp.trajectory_end):
            i -= 1
            i = i % self.experiences_stored
            exp = self.Exps[i]
        self.special_start = (i + 1) % self.experiences_stored
        self.trajectory_index = self.special_end

    def get_special_trajectory_indices(self, size):
        indices = []
        while len(indices) < size and self.trajectory_index != self.special_start:
            indices.append(self.trajectory_index)
            self.trajectory_index -= 1
            self.trajectory_index = self.trajectory_index % self.experiences_stored
        if self.trajectory_index == self.special_start:
            self.trajectory_index = self.special_end
        return indices

    # Sample an n-step target
    # Intended for use with step sizes of 1 for now
    def Sample_N(self, size, N, gamma, special_trajectory=False):
        assert(size <= self.N)
        if not special_trajectory:
            self.T += 1

        # indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        if special_trajectory:
            indices = self.get_special_trajectory_indices(size)
        else:
            indices = self.get_indices(size)

        self.Recompute_Pseudo_Counts(indices)
        batch_to_return = []
        for index in indices:
            exps_to_use = self.Exps[index: min(self.experiences_stored, index + N)]
            # Check for terminal states
            index_up_to = min(self.experiences_stored, index + N) - index
            for i, exp in enumerate(exps_to_use):
                # print(exp)
                if exp.terminal or exp.trajectory_end:
                    index_up_to = i + 1
                    break
            exps_to_use = exps_to_use[:index_up_to]
            # We then need to recompute the pseudo-counts for all of these
            # print([ii for ii in range(index, index + index_up_to)])
            self.Recompute_Pseudo_Counts([ii for ii in range(index, index + index_up_to)])

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
        return batch_to_return, indices

    def Sample_N_Eligibility_States(self, size, gamma, num_states=5, gap=2):
        assert(size <= self.N)
        self.T += 1
        N = gap ** (num_states)
        # indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        indices = self.get_indices(size)
        self.Recompute_Pseudo_Counts(indices)
        batch_to_return = []
        experiences_to_return = []
        for index in indices:
            exps_to_use = self.Exps[index: min(self.experiences_stored, index + N)]
            # Check for terminal states
            index_up_to = min(self.experiences_stored, index + N) - index
            for i, exp in enumerate(exps_to_use):
                # print(exp)
                if exp.terminal or exp.trajectory_end:
                    index_up_to = i + 1
                    break
            exps_to_use = exps_to_use[:index_up_to]

            # We then need to recompute the pseudo-counts for all of these
            # print([ii for ii in range(index, index + index_up_to)])
            self.Recompute_Pseudo_Counts([ii for ii in range(index, index + index_up_to)])

            state_now = exps_to_use[0].state
            action_now = exps_to_use[0].action
            # state_then = exps_to_use[-1].state_next
            terminate = exps_to_use[-1].terminal
            rewards_to_use = list(map(lambda x: x.reward + x.pseudo_reward, exps_to_use))
            # steps = len(exps_to_use)
            new_exp = (state_now, action_now, 0, None, 0, terminate)
            experiences_to_return.append(new_exp)

            # Work out the state we need a Q Value estimate for
            states_in_seq = []
            for i in [gap ** (m + 1) for m in range(num_states)]:
                i -= 1
                if i > len(exps_to_use) and len(states_in_seq) == 0:
                    i = len(exps_to_use) - 1
                if i < len(exps_to_use):
                    accum_reward = 0
                    for ri in reversed(rewards_to_use[:i]):
                        accum_reward = ri + gamma * accum_reward
                    states_in_seq.append((exps_to_use[i], accum_reward, i))

            if states_in_seq == []:
                print(new_exp)
                print(exps_to_use)
                print(len(exps_to_use))
            batch_to_return.append(states_in_seq)

        return batch_to_return, experiences_to_return
