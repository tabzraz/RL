import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from Models.Models import get_torch_models as get_models

from Replay.ExpReplay_Sarsa import ExperienceReplay_Sarsa as ExpReplay
from Replay.NEC_Replay import NEC_Replay as ExpReplay
from Replay.DND import DND


def kernel(h, h_i, delta=1e-3):
    return 1 / (torch.dist(h, h_i, p=2) + delta)


class NEC_Agent:

    def __init__(self, args, exp_model, logging_func):
        self.args = args

        # Exploration Model
        self.exp_model = exp_model

        # Experience Replay
        self.replay = ExpReplay(args.exp_replay_size, args)
        self.dnds = [DND(kernel=kernel, num_neighbors=50, max_memory=args.dnd_size, embedding_size=args.nec_embedding) for _ in range(self.args.actions)]

        # DQN and Target DQN
        model = get_models(args.model)
        self.embedding = model(embedding=args.nec_embedding)

        dqn_params = 0
        for weight in self.dqn.parameters():
            weight_params = 1
            for s in weight.size():
                weight_params *= s
            dqn_params += weight_params
        print("Embedding Network has {:,} parameters.".format(dqn_params))

        if args.gpu:
            print("Moving models to GPU.")
            self.embedding.cuda()

        # Optimizer
        self.optimizer = Adam(self.embedding.parameters(), lr=args.lr)

        self.T = 0
        self.target_sync_T = -self.args.t_max

        self.experiences = []

    def Q_Value_Estimates(self, state):

        # Get state embedding
        state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
        key = self.embedding(Variable(state, volatile=True)).cpu().data[0]

        estimate_from_dnds = [dnd.lookup(key) for dnd in self.dnds]
        return torch.cat(estimate_from_dnds), key
        # return np.array(estimate_from_dnds), key

    def act(self, state, epsilon, exp_model):

        q_values, key = self.Q_Value_Estimates(state)
        q_values_numpy = q_values.numpy()

        extra_info = {}
        extra_info["Q_Values"] = q_values_numpy

        if np.random.random() < epsilon:
            action = np.random.randint(low=0, high=self.args.actions)
        else:
            action = np.argmax(q_values_numpy)

        extra_info["Action"] = action

        return action, extra_info

    def experience(self, state, action, reward, state_next, steps, terminated, pseudo_reward=0, density=1, exploring=False):

        experience = (state, action, reward, pseudo_reward, state_next, terminated)
        self.experiences.append(experience)

        if len(self.experiences) >= self.args.n_step:
            first_state = self.experiences[0][0]
            first_action = self.experiences[0][1]
            last_state = self.experiences[-1][4]
            terminated_last_state = self.experiences[-1][5]
            accum_reward = 0
            for exp in reversed(self.experiences):
                r = exp[2]
                pr = exp[3]
                accum_reward += (r + pr) + self.args.gamma * accum_reward
            if terminated_last_state:
                last_state_max_q_val = 0
            else:
                last_state_q_val_estimates, last_state_key = self.Q_Value_Estimates(last_state)
                last_state_max_q_val = np.max(last_state_q_val_estimates)
            first_state_q_val_estimates, first_state_key = self.Q_Value_Estimates(first_state)

            n_step_q_val_estimate = accum_reward + (self.args.gamma ** self.args.n_step) * last_state_max_q_val

            # Add to dnd
            if self.dnds[first_action].is_present(key=first_state_key):
                current_q_val = self.dnds[first_action].get_value(key=first_state_key)
                new_q_val = current_q_val + self.nec_alpha(n_step_q_val_estimate - current_q_val)
                self.dnds[first_action].upsert(key=first_state_key, value=new_q_val)
            else:
                self.dnds[first_action].upsert(key=first_state_key, value=n_step_q_val_estimate)

            # Add to replay
            self.replay.Add_Exp(first_state, first_action, n_step_q_val_estimate)

            # Remove first experience
            self.experiences = self.experiences[1:]

        if not exploring:
            self.T += 1

    def end_of_trajectory(self):
        self.replay.end_of_trajectory()

        # Go through the experiences and add them to the replay using a less than N-step Q-Val estimate
        while len(self.experiences) > 0:
            first_state = self.experiences[0][0]
            first_action = self.experiences[0][1]
            last_state = self.experiences[-1][4]
            terminated_last_state = self.experiences[-1][5]
            accum_reward = 0
            for exp in reversed(self.experiences):
                r = exp[2]
                pr = exp[3]
                accum_reward += (r + pr) + self.args.gamma * accum_reward
            if terminated_last_state:
                last_state_max_q_val = 0
            else:
                last_state_q_val_estimates, last_state_key = self.Q_Value_Estimates(last_state)
                last_state_max_q_val = np.max(last_state_q_val_estimates)
            first_state_q_val_estimates, first_state_key = self.Q_Value_Estimates(first_state)

            n_step = len(self.experiences)
            n_step_q_val_estimate = accum_reward + (self.args.gamma ** n_step) * last_state_max_q_val

            # Add to dnd
            if self.dnds[first_action].is_present(key=first_state_key):
                current_q_val = self.dnds[first_action].get_value(key=first_state_key)
                new_q_val = current_q_val + self.nec_alpha(n_step_q_val_estimate - current_q_val)
                self.dnds[first_action].upsert(key=first_state_key, value=new_q_val)
            else:
                self.dnds[first_action].upsert(key=first_state_key, value=n_step_q_val_estimate)

            # Add to replay
            self.replay.Add_Exp(first_state, first_action, n_step_q_val_estimate)

            # Remove first experience
            self.experiences = self.experiences[1:]

    def train(self):

        info = {}

        for _ in range(self.args.iters):

            # TODO: Use a named tuple for experience replay
            batch = self.replay.Sample(self.args.batch_size)
            columns = list(zip(*batch))

            states = Variable(torch.from_numpy(np.array(columns[0])).float().transpose_(1, 3))
            actions = columns[1]
            targets = Variable(torch.FloatTensor(columns[2]))

            keys = 

            model_predictions = [self.dnds[action].lookup]
            predicted = torch.cat([self.dnd_list[sample.action].lookup(self.embedding_network(sample.state.float() / 255.0), True) for sample in batch]) 

            if self.args.gpu:
                actions = actions.cuda()
                q_value_targets = q_value_targets.cuda()
            model_predictions = self.dqn(states).gather(1, actions.view(-1, 1))

            # info = {}

            td_error = model_predictions - q_value_targets
            info["TD_Error"] = td_error.mean().data[0]

            # Update the priorities
            if not self.args.density_priority:
                self.replay.Update_Indices(indices, td_error.cpu().data.numpy(), no_pseudo_in_priority=self.args.count_td_priority)

            # If using prioritised we need to weight the td_error
            if self.args.prioritized and self.args.prioritized_is:
                # print(td_error)
                weights_tensor = torch.from_numpy(is_weights).float()
                weights_tensor = Variable(weights_tensor)
                if self.args.gpu:
                    weights_tensor = weights_tensor.cuda()
                # print(weights_tensor)
                td_error = td_error * weights_tensor
            l2_loss = (td_error).pow(2).mean()
            info["Loss"] = l2_loss.data[0]

            # Update
            self.optimizer.zero_grad()
            l2_loss.backward()

            # Taken from pytorch clip_grad_norm
            # Remove once the pip version it up to date with source
            gradient_norm = clip_grad_norm(self.dqn.parameters(), self.args.clip_value)
            if gradient_norm is not None:
                info["Norm"] = gradient_norm

            self.optimizer.step()

            if "States" in info:
                states_trained = info["States"]
                info["States"] = states_trained + columns[0]
            else:
                info["States"] = columns[0]

        self.replay.Clear()

        return info
