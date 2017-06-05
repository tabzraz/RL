import numpy as np
import collections
from .Binary_Heap import BinaryHeap


class ExperienceReplay_Options_Pseudo_Sort:

    Experience = collections.namedtuple("Experience", "state action reward state_next steps terminal pseudo_reward pseudo_reward_t trajectory_end")

    def __init__(self, N, pseudo_limit, exp_model, args, priority=False):
        self.N = N
        self.Exps = [None for _ in range(N)]
        self.pseudo_limit = pseudo_limit
        self.exp_model = exp_model
        self.T = 0
        self.storing_index = 0
        self.experiences_stored = 0
        self.args = args

        self.recent_exps = []

        self.priority = priority
        if self.priority:
            self.alpha = 0.7
            self.priorities = BinaryHeap(N)
            self.distrib = np.array([pow((1 / (i + 1)), self.alpha) for i in range(self.N)])
            self.full_distrib = self.distrib / np.sum(self.distrib)

    def Clear(self):
        self.Exps = [None for _ in range(self.N)]
        self.storing_index = 0
        self.experiences_stored = 0

    def Add_Exp(self, state_now, action, reward, state_after, steps, terminal, pseudo_reward=0):
        # if len(self.Exps) >= self.N:
            # self.Exps.pop(0)
        # Make copies just in case
        new_exp = self.Experience(state=np.copy(state_now), action=action, reward=reward, state_next=np.copy(state_after), steps=steps, terminal=terminal, pseudo_reward=pseudo_reward, pseudo_reward_t=self.T, trajectory_end=terminal)
        self.recent_exps.append(new_exp)
        if len(self.recent_exps) < self.args.n_step:
            return
        n_step_reward = 0
        n_step_psuedo_reward = 0
        for i in range(self.args.n_step):
            n_step_reward += (self.args.gamma ** i) * self.recent_exps[i].reward
            n_step_psuedo_reward += (self.args.gamma ** i) * self.recent_exps[i].pseudo_reward
        n_step_state = np.copy(self.recent_exps[0].state)
        n_step_action = self.recent_exps[0].action
        n_step_terminal = self.recent_exps[-1].terminal
        n_step_traj_end = self.recent_exps[-1].trajectory_end
        n_step_next_state = np.copy(self.recent_exps[-1].state_next)
        n_step_exp = self.Experience(state=n_step_state, action=n_step_action, reward=n_step_reward, state_next=n_step_next_state, steps=self.args.n_step, terminal=n_step_terminal, pseudo_reward=n_step_psuedo_reward, pseudo_reward_t=self.T, trajectory_end=n_step_traj_end)

        self.Exps[self.storing_index] = n_step_exp
        # Remove the oldest entry
        self.recent_exps = self.recent_exps[1:]

        if self.priority:
            p = self.priorities.get_max_priority()
            self.priorities.update(p, self.storing_index)
            if self.T % 1000 == 0:
                self.priorities.balance_tree()

        self.experiences_stored = max(self.experiences_stored, self.storing_index)

        if self.storing_index == self.N - 1:
            # We have stored the most experiences we can
            # Sort the list by count (pseudo-reward as proxy) now
            self.Exps.sort(key=lambda x: x.pseudo_reward, reverse=True)

        if self.storing_index < self.N - 1:
            self.storing_index += 1

    def end_of_trajectory(self):
        while len(self.recent_exps) > 0:
            n_step_reward = 0
            n_step_psuedo_reward = 0
            for i in range(len(self.recent_exps)):
                n_step_reward += (self.args.gamma ** i) * self.recent_exps[i].reward
                n_step_psuedo_reward += (self.args.gamma ** i) * self.recent_exps[i].pseudo_reward
            n_step_state = np.copy(self.recent_exps[0].state)
            n_step_action = self.recent_exps[0].action
            n_step_terminal = self.recent_exps[-1].terminal
            # n_step_traj_end = self.recent_exps[-1].trajectory_end
            n_step_next_state = np.copy(self.recent_exps[-1].state_next)
            n_step_exp = self.Experience(state=n_step_state, action=n_step_action, reward=n_step_reward, state_next=n_step_next_state, steps=len(self.recent_exps), terminal=n_step_terminal, pseudo_reward=n_step_psuedo_reward, pseudo_reward_t=self.T, trajectory_end=True)

            self.Exps[self.storing_index] = n_step_exp
            # Remove the oldest entry
            self.recent_exps = self.recent_exps[1:]

            if self.priority:
                p = self.priorities.get_max_priority()
                self.priorities.update(p, self.storing_index)
                if self.T % 1000 == 0:
                    self.priorities.balance_tree()

            self.experiences_stored = max(self.experiences_stored, self.storing_index)

            if self.storing_index == self.N - 1:
                # We have stored the most experiences we can
                # Sort the list by count (pseudo-reward as proxy) now
                self.Exps.sort(key=lambda x: x.pseudo_reward, reverse=True)

            if self.storing_index < self.N - 1:
                self.storing_index += 1

        # exp = self.Exps[self.storing_index - 1]
        # new_exp = list(exp)
        # new_exp[-1] = True  # End of trajectory
        # new_exp_tuple = self.Experience(*new_exp)
        # self.Exps[self.storing_index - 1] = new_exp_tuple

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
                if self.args.decay_count:
                    new_bonus = exp.pseudo_reward * self.args.decay_count_rate ** (self.T - exp.pseudo_reward_t)
                else:
                    new_bonus, _ = self.exp_model.bonus(exp.state, dont_remember=True)
                new_exp = self.Experience(state=exp.state, action=exp.action, reward=exp.reward, state_next=exp.state_next, steps=exp.steps, terminal=exp.terminal, pseudo_reward=new_bonus, pseudo_reward_t=self.T, trajectory_end=exp.trajectory_end)
                self.Exps[i] = new_exp

    def Update_Indices(self, indices, ps):
        if self.priority:
            # print(ps)
            for index, priority in zip(indices, ps):
                # print(index, priority)
                self.priorities.update(priority[0] + 0.00001, index)

    def Sample_N(self, size, N, gamma):
        assert(size <= self.N)
        self.T += 1
        # indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        indices = self.get_indices(size)
        self.Recompute_Pseudo_Counts(indices)
        batch_to_return = []

        # exps_to_use = self.Exps[indices]

        for exp in [self.Exps[i] for i in indices]:
            state_now = exp.state
            action_now = exp.action
            accum_reward = exp.reward + exp.pseudo_reward
            state_then = exp.state_next
            steps = exp.steps
            terminate = exp.terminal

        # state_now = exps_to_use[0].state
        # action_now = exps_to_use[0].action
        # state_then = exps_to_use[-1].state_next
        # terminate = exps_to_use[-1].terminal
        # steps = len(exps_to_use)
        # rewards_to_use = list(map(lambda x: x.reward + x.pseudo_reward, exps_to_use))
        # accum_reward = 0
        # for ri in reversed(rewards_to_use):
        #     accum_reward = ri + gamma * accum_reward
            new_exp = (state_now, action_now, accum_reward, state_then, steps, terminate)
            batch_to_return.append(new_exp)
        # [] for indices to match the prioritized replay
        return batch_to_return, indices
