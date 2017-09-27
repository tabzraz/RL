import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from Models.Models import get_torch_models as get_models

from Replay.NEC_Replay import NEC_Replay as ExpReplay
from Replay.DND import DND


def kernel(h, h_i, delta=1e-3):
    return 1 / (torch.pow(h - h_i, 2).sum() + delta)


class NEC_Agent:

    def __init__(self, args, exp_model, logging_func):
        self.args = args

        # Exploration Model
        self.exp_model = exp_model

        # Experience Replay
        self.replay = ExpReplay(args.exp_replay_size, args)
        self.dnds = [DND(kernel=kernel, num_neighbors=args.nec_neighbours, max_memory=args.dnd_size, embedding_size=args.nec_embedding) for _ in range(self.args.actions)]

        # DQN and Target DQN
        model = get_models(args.model)
        self.embedding = model(embedding=args.nec_embedding)

        embedding_params = 0
        for weight in self.embedding.parameters():
            weight_params = 1
            for s in weight.size():
                weight_params *= s
            embedding_params += weight_params
        print("Embedding Network has {:,} parameters.".format(embedding_params))

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
        key = self.embedding(Variable(state, volatile=True)).cpu()

        if (key != key).sum().data[0] > 0:
            pass
            # print(key)
            # for param in self.embedding.parameters():
                # print(param)
            # print(key != key)
            # print((key != key).sum().data[0])
            # print("Nan key")

        estimate_from_dnds = torch.cat([dnd.lookup(key) for dnd in self.dnds])
        # print(estimate_from_dnds)
        return estimate_from_dnds, key
        # return np.array(estimate_from_dnds), key

    def act(self, state, epsilon, exp_model):

        q_values, key = self.Q_Value_Estimates(state)
        q_values_numpy = q_values.data.numpy()

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
            self.add_experience()

        if not exploring:
            self.T += 1

    def end_of_trajectory(self):
        self.replay.end_of_trajectory()

        # Go through the experiences and add them to the replay using a less than N-step Q-Val estimate
        while len(self.experiences) > 0:
            self.add_experience()

    def add_experience(self):
        first_state = self.experiences[0][0]
        first_action = self.experiences[0][1]
        last_state = self.experiences[-1][4]
        terminated_last_state = self.experiences[-1][5]
        accum_reward = 0
        for ex in reversed(self.experiences):
            r = ex[2]
            pr = ex[3]
            accum_reward = (r + pr) + self.args.gamma * accum_reward
        # if accum_reward > 1000:
            # print(accum_reward)
        if terminated_last_state:
            last_state_max_q_val = 0
        else:
            last_state_q_val_estimates, last_state_key = self.Q_Value_Estimates(last_state)
            last_state_max_q_val = last_state_q_val_estimates.data.max(0)[0][0]
            # print(last_state_max_q_val)
        first_state_q_val_estimates, first_state_key = self.Q_Value_Estimates(first_state)
        first_state_key = first_state_key.data[0].numpy()

        n_step_q_val_estimate = accum_reward + (self.args.gamma ** len(self.experiences)) * last_state_max_q_val
        n_step_q_val_estimate = n_step_q_val_estimate
        # print(n_step_q_val_estimate)

        # Add to dnd
        # print(first_state_key)
        # print(tuple(first_state_key.data[0]))
        # if any(np.isnan(first_state_key)):
            # print("NAN")
        if self.dnds[first_action].is_present(key=first_state_key):
            current_q_val = self.dnds[first_action].get_value(key=first_state_key)
            new_q_val = current_q_val + self.args.nec_alpha * (n_step_q_val_estimate - current_q_val)
            self.dnds[first_action].upsert(key=first_state_key, value=new_q_val)
        else:
            self.dnds[first_action].upsert(key=first_state_key, value=n_step_q_val_estimate)

        # Add to replay
        self.replay.Add_Exp(first_state, first_action, n_step_q_val_estimate)

        # Remove first experience
        self.experiences = self.experiences[1:]

    def train(self):

        info = {}
        if self.T % self.args.nec_update != 0:
            return info

        # print("Training")

        for _ in range(self.args.iters):

            # TODO: Use a named tuple for experience replay
            batch = self.replay.Sample(self.args.batch_size)
            columns = list(zip(*batch))

            states = Variable(torch.from_numpy(np.array(columns[0])).float().transpose_(1, 3))
            # print(states)
            actions = columns[1]
            # print(actions)
            targets = Variable(torch.FloatTensor(columns[2]))
            # print(targets)
            keys = self.embedding(states).cpu()
            # print(keys)
            # print("Keys", keys.requires_grad)
            # for action in actions:
                # print(action)
            # for action, key in zip(actions, keys):
                # print(action, key)
                # kk = key.unsqueeze(0)
                # print("kk", kk.requires_grad)
                # k = self.dnds[action].lookup(key.unsqueeze(0))
                # print("key", key.requires_grad, key.volatile)
            model_predictions = torch.cat([self.dnds[action].lookup(key.unsqueeze(0)) for action, key in zip(actions, keys)])
            # print(model_predictions)
            # print(targets)

            td_error = model_predictions - targets
            # print(td_error)
            info["TD_Error"] = td_error.mean().data[0]

            l2_loss = (td_error).pow(2).mean()
            info["Loss"] = l2_loss.data[0]

            # Update
            self.optimizer.zero_grad()

            l2_loss.backward()

            # Taken from pytorch clip_grad_norm
            # Remove once the pip version it up to date with source
            gradient_norm = clip_grad_norm(self.embedding.parameters(), self.args.clip_value)
            if gradient_norm is not None:
                info["Norm"] = gradient_norm

            self.optimizer.step()

            if "States" in info:
                states_trained = info["States"]
                info["States"] = states_trained + columns[0]
            else:
                info["States"] = columns[0]

        return info
