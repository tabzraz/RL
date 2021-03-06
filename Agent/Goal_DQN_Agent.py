import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from Replay.ExpReplay_Goal_State import ExperienceReplay_Goal_State as ExpReplay
from Models.Models import get_torch_models as get_models
import os


class Goal_DQN_Agent:

    def __init__(self, args, exp_model, logging_funcs):
        self.args = args

        self.log = logging_funcs["log"]
        self.log_image = logging_funcs["image"]
        os.makedirs("{}/goal_states".format(args.log_path))
        self.env = logging_funcs["env"]

        # Exploration Model
        self.exp_model = exp_model

        # Experience Replay
        self.replay = ExpReplay(args.exp_replay_size, args.stale_limit, exp_model, args, priority=self.args.prioritized)

        # DQN and Target DQN
        model = get_models(args.model)
        self.model = model
        self.dqn = model(actions=args.actions)
        self.target_dqn = model(actions=args.actions)

        dqn_params = 0
        for weight in self.dqn.parameters():
            weight_params = 1
            for s in weight.size():
                weight_params *= s
            dqn_params += weight_params
        print("DQN has {:,} parameters.".format(dqn_params))

        self.target_dqn.eval()

        if args.gpu:
            print("Moving models to GPU.")
            self.dqn.cuda()
            self.target_dqn.cuda()

        # Optimizer
        self.optimizer = RMSprop(self.dqn.parameters(), lr=args.lr)
        # self.optimizer = Adam(self.dqn.parameters(), lr=args.lr)

        self.T = 0
        self.target_sync_T = -self.args.t_max

        # Hierarhcical stuff
        self.goal_state_T = 0
        self.goal_state = None

        self.max_bonus = 0

        self.goal_optimizer = None

        self.goal_dqn = None
        self.goal_dqns = []
        self.goal_dqn_states = []

        self.executing_option = False
        self.option_num = 0
        self.option_steps = 0

        self.training_goal_T = 0

    def sync_target_network(self):
        for target, source in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target.data = source.data

    def option_act(self, state):
        current_goal_dqn = self.goal_dqns[self.option_num]
        current_goal_dqn.eval()
        state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
        q_values = current_goal_dqn(Variable(state, volatile=True)).cpu().data[0]
        q_values_numpy = q_values.numpy()

        action = q_values.max(0)[1][0]  # Torch...

        self.option_steps += 1

        if self.option_steps >= self.args.max_option_steps:
            print("Finishing execution of Goal DQN {}: Max Steps Reached".format(self.option_num))
            self.log("Option_{}/Steps_Max".format(self.option_num), self.option_steps, step=self.T)
            self.option_num += 1
            self.option_steps = 0
            if self.option_num == len(self.goal_dqns):
                self.executing_option = False
                self.option_num = 0

        extra_info = {}
        extra_info["Q_Values"] = q_values_numpy
        extra_info["Action"] = action

        return action, extra_info

    def act(self, state, epsilon, exp_model):
        # self.T += 1

        if self.executing_option and np.allclose(state, self.goal_dqn_states[self.option_num]):
            print("Finishing execution of Goal DQN {}: Goal Reached".format(self.option_num))
            self.log("Option_{}/Steps_Goal".format(self.option_num), self.option_steps, step=self.T)
            self.option_num += 1
            self.option_steps = 0
            if self.option_num == len(self.goal_dqns):
                self.executing_option = False
                self.option_num = 0

        if self.executing_option:
            return self.option_act(state)

        self.dqn.eval()
        orig_state = state[:, :, -1:]
        state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
        q_values = self.dqn(Variable(state, volatile=True)).cpu().data[0]
        q_values_numpy = q_values.numpy()

        extra_info = {}
        extra_info["Q_Values"] = q_values_numpy

        if self.args.optimistic_init:
            if not self.args.ucb:
                for a in range(self.args.actions):
                    _, info = exp_model.bonus(orig_state, a, dont_remember=True)
                    action_pseudo_count = info["Pseudo_Count"]
                    # TODO: Log the optimism bonuses
                    q_values[a] += self.args.optimistic_scaler / np.sqrt(action_pseudo_count + 0.01)
            else:
                action_counts = []
                for a in range(self.args.actions):
                    _, info = exp_model.bonus(orig_state, a, dont_remember=True)
                    action_pseudo_count = info["Pseudo_Count"]
                    action_counts.append(action_pseudo_count)
                total_count = sum(action_counts)
                for ai, a in enumerate(action_counts):
                    # TODO: Log the optimism bonuses
                    q_values[ai] += self.args.optimistic_scaler * np.sqrt(2 * np.log(max(1, total_count)) / (a + 0.01))

        if np.random.random() < epsilon:
            action = np.random.randint(low=0, high=self.args.actions)
        else:
            action = q_values.max(0)[1][0]  # Torch...

        if self.args.force_low_count_action:
            # Calculate the counts for each actions
            for a in range(self.args.actions):
                _, info = exp_model.bonus(orig_state, a, dont_remember=True)
                action_pseudo_count = info["Pseudo_Count"]
                # Pick the first one out of simplicity
                if action_pseudo_count < self.args.min_action_count:
                    action = a
                    extra_info["Forced_Action"] = a
                    break

        extra_info["Action"] = action

        return action, extra_info

    def experience(self, state, action, reward, state_next, steps, terminated, pseudo_reward=0, density=1, exploring=False):
        if not exploring:
            self.T += 1
        self.replay.Add_Exp(state, action, reward, state_next, steps, terminated, pseudo_reward, density)

        self.max_bonus = max(pseudo_reward, self.max_bonus)
        if pseudo_reward >= self.args.goal_state_threshold * self.max_bonus and self.T - self.goal_state_T > self.args.goal_state_interval:
            self.goal_state = state
            self.goal_state_image = self.env.state_to_image(self.goal_state)
            self.train_goal_network()

    def train_goal_network(self):
        # Make a copy of the dqn and reinit the last weights
        self.goal_dqn = self.model(actions=self.args.actions)
        self.goal_dqn.load_state_dict(self.dqn.state_dict())
        fc_features = self.goal_dqn.qvals.in_features
        self.goal_dqn.qvals = nn.Linear(fc_features, self.args.actions)
        if self.args.gpu:
            print("Moving goal_dqn to GPU")
            self.goal_dqn.cuda()

        self.goal_optimizer = RMSprop(self.goal_dqn.parameters(), lr=self.args.lr)
        # self.goal_optimizer = Adam(self.goal_dqn.parameters(), lr=self.args.lr)

        option_num = len(self.goal_dqns)
        print("Training Goal DQN {}".format(option_num))
        for _ in range(self.args.goal_iters):
            self.training_goal_T += 1

            train_info = self.train(goal_train=True)

            self.log("Option_{}/Gradient_Norm".format(option_num), train_info["Norm"], step=self.training_goal_T)
            self.log("Option_{}/Loss".format(option_num), train_info["Loss"], step=self.training_goal_T)
            self.log("Option_{}/TD_Error".format(option_num), train_info["TD_Error"], step=self.training_goal_T)
        # print("Finished training Goal DQN {}\n\n\n\n".format(len(self.goal_dqns)))

        self.goal_dqns.append(self.goal_dqn)
        self.goal_dqn_states.append(self.goal_state)
        # print(self.goal_state)
        # print(self.goal_dqn_states[-1].shape)

        # Save the goal state
        self.log_image("{}/goal_states/Goal_State_{}".format(self.args.log_path, option_num), self.goal_state_image)

        self.goal_state_T = self.T
        # print("Goal State T is :", self.goal_state_T)

    def end_of_trajectory(self):
        self.replay.end_of_trajectory()

        if len(self.goal_dqns) > 0:
            print("Gonna start executing")
            self.executing_option = True
            self.option_num = 0
        self.option_steps = 0

    def train(self, goal_train=False):
        if self.T - self.target_sync_T > self.args.target:
            self.sync_target_network()
            self.target_sync_T = self.T

        info = {}

        for _ in range(self.args.iters):
            if goal_train:
                dqn = self.goal_dqn
            else:
                dqn = self.dqn
            dqn.eval()

            n_step_sample = self.args.n_step
            if goal_train:
                batch, indices, is_weights = self.replay.Sample_N_GoalState(self.args.batch_size, n_step_sample, self.args.gamma, self.goal_state)
            else:
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

            if goal_train:
                target_dqn_qvals = dqn(new_states).cpu()
            else:
                target_dqn_qvals = self.target_dqn(new_states).cpu()
            # Make a new variable with those values so that these are treated as constants
            target_dqn_qvals_data = Variable(target_dqn_qvals.data)

            q_value_targets = (Variable(torch.ones(terminal_states.size()[0])) - terminal_states)
            inter = Variable(torch.ones(terminal_states.size()[0]) * self.args.gamma)
            # print(steps)
            q_value_targets = q_value_targets * torch.pow(inter, steps)
            if self.args.double:
                # Double Q Learning
                new_states_qvals = dqn(new_states).cpu()
                new_states_qvals_data = Variable(new_states_qvals.data)
                q_value_targets = q_value_targets * target_dqn_qvals_data.gather(1, new_states_qvals_data.max(1)[1])
            else:
                q_value_targets = q_value_targets * target_dqn_qvals_data.max(1)[0]
            q_value_targets = q_value_targets + rewards

            dqn.train()
            if self.args.gpu:
                actions = actions.cuda()
                q_value_targets = q_value_targets.cuda()
            model_predictions = dqn(states).gather(1, actions.view(-1, 1))

            # info = {}

            td_error = model_predictions - q_value_targets
            info["TD_Error"] = td_error.mean().data[0]

            # Update the priorities
            if not self.args.density_priority and not goal_train:
                self.replay.Update_Indices(indices, td_error.cpu().data.numpy(), no_pseudo_in_priority=self.args.count_td_priority)

            # If using prioritised we need to weight the td_error
            if self.args.prioritized and self.args.prioritized_is and not goal_train:
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
            optimizer = self.optimizer
            if goal_train:
                optimizer = self.goal_optimizer
            optimizer.zero_grad()
            l2_loss.backward()

            # Taken from pytorch clip_grad_norm
            # Remove once the pip version it up to date with source
            gradient_norm = clip_grad_norm(dqn.parameters(), self.args.clip_value)
            if gradient_norm is not None:
                info["Norm"] = gradient_norm

            optimizer.step()

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
