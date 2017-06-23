import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from Replay.ExpReplay_Options_Pseudo import ExperienceReplay_Options_Pseudo as ExpReplay
from Replay.ExpReplay_Options_Pseudo_Set import ExperienceReplay_Options_Pseudo_Set as ExpReplaySet
from Models.Models import get_torch_models as get_models


class TabQ_Agent:

    def __init__(self, args, exp_model):
        self.args = args

        # Exploration Model
        self.exp_model = exp_model

        # Experience Replay
        if self.args.set_replay:
            self.replay = ExpReplaySet(10, 10, exp_model, args, priority=False)
        else:
            self.replay = ExpReplay(args.exp_replay_size, args.stale_limit, exp_model, args, priority=self.args.prioritized)

        # DQN and Target DQN
        self.q_table = {}
        self.actions = args.actions

        self.T = 0

    def state_to_tuple(self, state):
        return tuple(np.argwhere(state > 0.9)[0])
        return tuple([tuple([tuple(y) for y in x]) for x in state])

    def act(self, state, epsilon, exp_model):
        self.T += 1

        tuple_state = self.state_to_tuple(state)
        q_values = []
        for a in range(self.actions):
            key = (tuple_state, a)
            if key in self.q_table:
                q_value = self.q_table[key]
            else:
                q_value = np.random.random() / 1000
            q_values.append(q_value)

        q_values_numpy = np.array(q_values)

        extra_info = {}
        extra_info["Q_Values"] = q_values_numpy

        if self.args.optimistic_init:
            for a in range(self.args.actions):
                _, info = exp_model.bonus(state, a, dont_remember=True)
                action_pseudo_count = info["Pseudo_Count"]
                # TODO: Log the optimism bonuses
                q_values_numpy[a] += self.args.optimistic_scaler / np.sqrt(action_pseudo_count + 0.01)

        if np.random.random() < epsilon:
            action = np.random.randint(low=0, high=self.args.actions)
        else:
            action = q_values_numpy.argmax()

        extra_info["Action"] = action

        return action, extra_info

    def experience(self, state, action, reward, state_next, steps, terminated, pseudo_reward=0):
        self.replay.Add_Exp(state, action, reward, state_next, steps, terminated, pseudo_reward)

    def end_of_trajectory(self):
        self.replay.end_of_trajectory()

    def train(self):

        for _ in range(self.args.iters):

            # TODO: Use a named tuple for experience replay
            n_step_sample = 1
            if np.random.random() < self.args.n_step_mixing:
                n_step_sample = self.args.n_step
            batch, indices, is_weights = self.replay.Sample_N(self.args.batch_size, n_step_sample, self.args.gamma)

            # (state_now, action_now, accum_reward, state_then, steps, terminate)
            td_errors = []
            for index, batch_stuff in enumerate(batch):

                state, action, reward, next_state, step, terminal_state = batch_stuff
                # Work out q_value target
                q_value_target = reward
                if not terminal_state:

                    next_state_max_qval = -100000
                    tuple_next_state = self.state_to_tuple(next_state)
                    for a in range(self.actions):
                        key = (tuple_next_state, a)
                        if key in self.q_table:
                            q_value = self.q_table[key]
                        else:
                            # q_value = 0
                            q_value = np.random.random() / 1000
                        next_state_max_qval = max(next_state_max_qval, q_value)

                    q_value_target += (self.args.gamma ** step) * next_state_max_qval

                # Update our estimate
                tuple_state = self.state_to_tuple(state)
                q_value_prediction = 0
                key = (tuple_state, action)
                if key in self.q_table:
                    q_value_prediction = self.q_table[key]

                td_error = q_value_prediction - q_value_target
                td_errors.append(td_error)

                if self.args.prioritized and self.args.prioritized_is:
                    td_error * is_weights[index]

                updated_prediction = q_value_prediction - self.args.lr * (td_error)

                self.q_table[key] = updated_prediction

            info = {}

            info["TD_Error"] = sum(td_errors) / len(td_errors)

            # Update the priorities
            if not self.args.bonus_priority:
                self.replay.Update_Indices(indices, td_errors, no_pseudo_in_priority=self.args.count_td_priority)

            info["Loss"] = info["TD_Error"]

        return info
