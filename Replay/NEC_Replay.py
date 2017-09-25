import numpy as np
import collections
from .Binary_Heap import BinaryHeap
import sys
import random


class ExperienceReplay_Sarsa:

    Experience = collections.namedtuple("Experience", "state action estimate")

    def __init__(self, N, args):
        self.N = N
        self.args = args
        self.Exps = [None for _ in range(N)]

        self.T = 0
        self.storing_index = 0
        self.experiences_stored = 0

        self.printed_ram_usage = False

    def Clear(self):
        self.Exps = [None for _ in range(self.N)]
        self.storing_index = 0
        self.experiences_stored = 0

    def Add_Exp(self, state_now, action, q_val_estimate):
        if self.storing_index >= self.N:
            self.storing_index = 0
        # if len(self.Exps) >= self.N:
            # self.Exps.pop(0)
        # Make copies just in case, make them float32 to save on ram
        state_now = state_now.astype(np.float32)
        new_exp = self.Experience(state=np.copy(state_now), action=action, estimate=q_val_estimate)
        self.Exps[self.storing_index] = new_exp

        if not self.printed_ram_usage:
            exps_size = sys.getsizeof(self.Exps) / (1024.0)
            state_size = np.copy(state_now).nbytes / (1024.0 ** 2)
            print("\n\nState is of size roughly {:.2f} MB".format(state_size))
            mb_size = state_size * 2 * self.N
            mb_size += exps_size
            print("{:,} Experiences will take at least {:.2f} GB\n".format(self.N, mb_size / 1024))
            self.printed_ram_usage = True

        self.storing_index += 1
        self.experiences_stored = max(self.experiences_stored, self.storing_index)

    def end_of_trajectory(self):
        pass

    def get_sampling_distribution(self):
        if self.experiences_stored < self.N:
            partial = self.distrib[:self.experiences_stored]
            return partial / np.sum(partial)
        else:
            return self.full_distrib

    def get_indices(self, size):
        indices = np.random.randint(low=0, high=self.experiences_stored, size=size)
        return indices

    def Recompute_Pseudo_Counts(self, indices):
        if self.exp_model is None or self.pseudo_limit >= self.N:
            # print("No exp model")
            return
        # ps = []
        for i in indices:
            exp = self.Exps[i]
            if self.T - exp.pseudo_reward_t > self.pseudo_limit:
                # Recompute it
                new_bonus, new_bonus_info = self.exp_model.bonus(exp.state, dont_remember=True)
                density = new_bonus_info["Density"]
                new_exp = self.Experience(state=exp.state, action=exp.action, reward=exp.reward, state_next=exp.state_next, steps=exp.steps, terminal=exp.terminal, pseudo_reward=new_bonus, density=density, pseudo_reward_t=self.T, trajectory_end=exp.trajectory_end)
                self.Exps[i] = new_exp
                self.densities[i] = 1 / density
                # if self.args.bonus_priority:
                #     ps.append((1 / density))

        # if self.args.bonus_priority:
        #     self.Update_Indices(indices, ps)

    def Update_Indices(self, indices, ps, no_pseudo_in_priority=False):
        if self.priority:
            # print(ps)
            # if self.pseudo_rewards_used is None:
                # self.pseudo_rewards_used = [0 for i in indices]
            for index, priority, pseudo_reward in zip(indices, ps, self.pseudo_rewards_used):
                # print(index, priority, pseudo_reward)
                priority_value = priority
                if priority < 0:
                    priority_value *= self.args.negative_td_scaler
                abs_priority = abs(priority_value)
                if no_pseudo_in_priority:
                    # TD Error is (Prediction - Target)
                    abs_priority -= abs(pseudo_reward) * self.args.count_td_scaler
                    if abs_priority < 0:
                        abs_priority = 0
                self.priorities.update(abs_priority + 0.00001, index)

    def Sample(self, size):
        # assert(size <= self.N)
        # if len(self.Exps) < size:
            # return self.Exps
        # indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        self.T += 1
        indices = self.get_indices(size)
        # self.Recompute_Pseudo_Counts(indices)
        # sample = [self.Exps[i] for i in indices]
        return [self.Exps[i] for i in indices]

    # Sample an n-step target
    # Intended for use with step sizes of 1 for now
    def Sample_N(self, size, N, gamma):
        assert(size <= self.N)
        self.T += 1
        # indices = np.random.randint(low=0, high=len(self.Exps) - 1, size=size)
        indices, is_weights = self.get_indices(size)
        self.Recompute_Pseudo_Counts(indices)
        batch_to_return = []
        pseudo_rewards_used = []
        for index in indices:
            N_Step = N
            if self.args.variable_n_step:
                exp_reward = self.Exps[index].pseudo_reward + self.Exps[index].reward
                if exp_reward > 0:
                    N_Step = min(N, int(self.max_reward / exp_reward))
                else:
                    N_Step = N
                N_Step = max(N_Step, 1)
                # print(N_Step)
            exps_to_use = self.Exps[index: min(self.experiences_stored, index + N_Step)]
            # Check for terminal states
            index_up_to = min(self.experiences_stored, index + N_Step) - index
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
            action_then = exps_to_use[-1].action_next
            terminate = exps_to_use[-1].terminal
            steps = len(exps_to_use)
            rewards_to_use = list(map(lambda x: x.reward + x.pseudo_reward, exps_to_use))
            pseudo_rewards = list(map(lambda x: x.pseudo_reward, exps_to_use))
            accum_reward = 0
            accum_psuedo_reward = 0
            for ri, pri in zip(reversed(rewards_to_use), reversed(pseudo_rewards)):
                accum_reward = ri + gamma * accum_reward
                accum_psuedo_reward = pri + gamma * accum_psuedo_reward
            new_exp = (state_now, action_now, accum_reward, state_then, steps, terminate, action_then)
            batch_to_return.append(new_exp)
            pseudo_rewards_used.append(accum_psuedo_reward)
        # [] for indices to match the prioritized replay

        # Remeber the pseudo rewards used
        self.pseudo_rewards_used = pseudo_rewards_used
        return batch_to_return, indices, is_weights
