import numpy as np
import torch
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from Replay.ExpReplay_Options_Pseudo import ExperienceReplay_Options_Pseudo as ExpReplay
from Models.Models import get_torch_models as get_models
import os


class DQN_Model_Agent:

    def __init__(self, args, exp_model, logging_func):
        self.args = args

        # Exploration Model
        self.exp_model = exp_model

        self.log = logging_func["log"]
        self.log_image = logging_func["image"]
        os.makedirs("{}/transition_model".format(args.log_path))

        # Experience Replay
        self.replay = ExpReplay(args.exp_replay_size, args.stale_limit, exp_model, args, priority=self.args.prioritized)

        # DQN and Target DQN
        model = get_models(args.model)
        print("\n\nDQN")
        self.dqn = model(actions=args.actions)
        print("Target DQN")
        self.target_dqn = model(actions=args.actions)

        dqn_params = 0
        for weight in self.dqn.parameters():
            weight_params = 1
            for s in weight.size():
                weight_params *= s
            dqn_params += weight_params
        print("Model DQN has {:,} parameters.".format(dqn_params))

        self.target_dqn.eval()

        if args.gpu:
            print("Moving models to GPU.")
            self.dqn.cuda()
            self.target_dqn.cuda()

        # Optimizer
        # self.optimizer = Adam(self.dqn.parameters(), lr=args.lr)
        self.optimizer = RMSprop(self.dqn.parameters(), lr=args.lr)

        self.T = 0
        self.target_sync_T = -self.args.t_max

        # Action sequences
        self.actions_to_take = []

    def sync_target_network(self):
        for target, source in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target.data = source.data

    def act(self, state, epsilon, exp_model, evaluation=False):
        # self.T += 1
        if not evaluation:
            if len(self.actions_to_take) > 0:
                action_to_take = self.actions_to_take[0]
                self.actions_to_take = self.actions_to_take[1:]
                return action_to_take, {}

        self.dqn.eval()
        orig_state = state[:, :, -1:]
        state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
        q_values = self.dqn(Variable(state, volatile=True)).cpu().data[0]
        q_values_numpy = q_values.numpy()

        extra_info = {}

        if self.args.optimistic_init and not evaluation and len(self.actions_to_take) == 0:

            # 2 action lookahead
            optimistic_estimates = []
            for action_1 in range(self.args.actions):
                second_action_estimates = []

                one_hot_action_1 = torch.zeros(1, self.args.actions)
                one_hot_action_1[0, action_1] = 1
                first_state_qvals, next_state_prediction = self.dqn(Variable(state, volatile=True), Variable(one_hot_action_1, volatile=True))
                first_state_qvals = first_state_qvals.cpu()
                next_state_prediction = next_state_prediction.cpu()

                next_state_qvals = self.dqn(next_state_prediction).cpu()
                next_state_qvals = next_state_qvals.data[0].numpy()

                first_state_qvals = first_state_qvals.data[0].numpy()

                action_1_reward = first_state_qvals[action_1] - self.args.gamma * np.argmax(next_state_qvals)

                numpy_state = state[0].numpy()
                numpy_state = np.swapaxes(numpy_state, 0, 2)
                _, info = exp_model.bonus(numpy_state, action_1, dont_remember=True)
                action_1_pseudo_count = info["Pseudo_Count"]
                action_1_bonus = self.args.optimistic_scaler / np.sqrt(action_1_pseudo_count + 0.01)

                next_state_numpy = next_state_prediction.data[0].numpy()
                next_state_numpy = np.swapaxes(next_state_numpy, 0, 2)

                for action_2 in range(self.args.actions):
                    action_2_q_val = next_state_qvals[action_2]

                    _, info = exp_model.bonus(next_state_numpy, action_2, dont_remember=True)
                    action_2_pseudo_count = info["Pseudo_Count"]
                    action_2_bonus = self.args.optimistic_scaler / np.sqrt(action_2_pseudo_count + 0.01)

                    seq_optimistic_estimate = action_1_reward + action_1_bonus + self.args.gamma * (action_2_q_val + action_2_bonus)
                    # print(action_1_reward, action_1_bonus)
                    # print("Esimate",seq_optimistic_estimate)
                    second_action_estimates.append(seq_optimistic_estimate)

                # print(second_action_estimates)
                optimistic_estimates.append(second_action_estimates)

            # print(optimistic_estimates)
            # Find the maximum sequence 
            max_so_far = -100000
            best_seq = [0, 0]
            for action_1 in range(self.args.actions):
                for action_2 in range(self.args.actions):
                    # print(optimistic_estimates[action_1], optimistic_estimates[action_1][action_2])
                    if optimistic_estimates[action_1][action_2] > max_so_far:
                        max_so_far = optimistic_estimates[action_1][action_2]
                        best_seq = [action_1, action_2]

            self.actions_to_take = best_seq

        extra_info["Q_Values"] = q_values_numpy

        if np.random.random() < epsilon:
            action = np.random.randint(low=0, high=self.args.actions)
        else:
            action = q_values.max(0)[1][0]  # Torch...

        extra_info["Action"] = action

        return action, extra_info

    def experience(self, state, action, reward, state_next, steps, terminated, pseudo_reward=0, density=1, exploring=False):
        if not exploring:
            self.T += 1
        self.replay.Add_Exp(state, action, reward, state_next, steps, terminated, pseudo_reward, density)

    def end_of_trajectory(self):
        self.replay.end_of_trajectory()

    def train(self):

        if self.T - self.target_sync_T > self.args.target:
            self.sync_target_network()
            self.target_sync_T = self.T

        info = {}

        for _ in range(self.args.iters):
            self.dqn.eval()

            # TODO: Use a named tuple for experience replay
            n_step_sample = self.args.n_step
            batch, indices, is_weights = self.replay.Sample_N(self.args.batch_size, n_step_sample, self.args.gamma)
            columns = list(zip(*batch))

            states = Variable(torch.from_numpy(np.array(columns[0])).float().transpose_(1, 3))
            actions = Variable(torch.LongTensor(columns[1]))
            terminal_states = Variable(torch.FloatTensor(columns[5]))
            rewards = Variable(torch.FloatTensor(columns[2]))
            # Have to clip rewards for DQN
            rewards = torch.clamp(rewards, -1, 1)
            steps = Variable(torch.FloatTensor(columns[4]))
            new_states = Variable(torch.from_numpy(np.array(columns[3])).float().transpose_(1, 3))

            target_dqn_qvals = self.target_dqn(new_states).cpu()
            # Make a new variable with those values so that these are treated as constants
            target_dqn_qvals_data = Variable(target_dqn_qvals.data)

            q_value_targets = (Variable(torch.ones(terminal_states.size()[0])) - terminal_states)
            inter = Variable(torch.ones(terminal_states.size()[0]) * self.args.gamma)
            # print(steps)
            q_value_targets = q_value_targets * torch.pow(inter, steps)
            if self.args.double:
                # Double Q Learning
                new_states_qvals = self.dqn(new_states).cpu()
                new_states_qvals_data = Variable(new_states_qvals.data)
                q_value_targets = q_value_targets * target_dqn_qvals_data.gather(1, new_states_qvals_data.max(1)[1])
            else:
                q_value_targets = q_value_targets * target_dqn_qvals_data.max(1)[0]
            q_value_targets = q_value_targets + rewards

            self.dqn.train()

            one_hot_actions = torch.zeros(self.args.batch_size, self.args.actions)

            for i in range(self.args.batch_size):
                one_hot_actions[i][actions[i].data] = 1

            if self.args.gpu:
                actions = actions.cuda()
                one_hot_actions = one_hot_actions.cuda()
                q_value_targets = q_value_targets.cuda()
                new_states = new_states.cuda()
            model_predictions_q_vals, model_predictions_state = self.dqn(states, Variable(one_hot_actions))
            model_predictions = model_predictions_q_vals.gather(1, actions.view(-1, 1))

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

            # Model 1 step state transition error

            # Save them every x steps
            if self.T % self.args.model_save_image == 0:
                os.makedirs("{}/transition_model/{}".format(self.args.log_path, self.T))
                for ii, image, action, next_state, current_state in zip(range(self.args.batch_size), model_predictions_state.cpu().data, actions.data, new_states.cpu().data, states.cpu().data):
                    image = image.numpy()[0]
                    image = np.clip(image, 0, 1)
                    # print(next_state)
                    next_state = next_state.numpy()[0]
                    current_state = current_state.numpy()[0]

                    black_bars = np.zeros_like(next_state[:1, :])
                    # print(black_bars.shape)

                    joined_image = np.concatenate((current_state, black_bars, image, black_bars, next_state), axis=0)
                    joined_image = np.transpose(joined_image)
                    self.log_image("{}/transition_model/{}/{}_____Action_{}".format(self.args.log_path, self.T, ii + 1, action), joined_image * 255)

                    # self.log_image("{}/transition_model/{}/{}_____Action_{}".format(self.args.log_path, self.T, ii + 1, action), image * 255)
                    # self.log_image("{}/transition_model/{}/{}_____Correct".format(self.args.log_path, self.T, ii + 1), next_state * 255)

            # print(model_predictions_state)

            # Cross Entropy Loss
            # TODO

            # Regresssion loss
            state_error = model_predictions_state - new_states
            # state_error_val = state_error.mean().data[0]

            info["State_Error"] = state_error.mean().data[0]
            self.log("DQN/State_Loss", state_error.mean().data[0], step=self.T)
            self.log("DQN/State_Loss_Squared", state_error.pow(2).mean().data[0], step=self.T)
            self.log("DQN/State_Loss_Max", state_error.abs().max().data[0], step=self.T)
            # self.log("DQN/Action_Matrix_Norm", self.dqn.action_matrix.weight.norm().cpu().data[0], step=self.T)

            combined_loss = (1 - self.args.model_loss) * td_error.pow(2).mean() + (self.args.model_loss) * state_error.pow(2).mean()
            l2_loss = combined_loss
            # l2_loss = (combined_loss).pow(2).mean()
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

        # Pad out the states to be of size batch_size
        if len(info["States"]) < self.args.batch_size:
            old_states = info["States"]
            new_states = old_states[0] * (self.args.batch_size - len(old_states))
            info["States"] = new_states

        return info
