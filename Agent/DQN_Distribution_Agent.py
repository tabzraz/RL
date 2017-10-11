import numpy as np
import torch
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from Replay.ExpReplay_Options_Pseudo import ExperienceReplay_Options_Pseudo as ExpReplay
from Models.Models import get_torch_models as get_models


class DQN_Distribution_Agent:

    def __init__(self, args, exp_model, logging_func):
        self.args = args

        # Exploration Model
        self.exp_model = exp_model

        self.log = logging_func["log"]

        # Experience Replay
        self.replay = ExpReplay(args.exp_replay_size, args.stale_limit, exp_model, args, priority=self.args.prioritized)

        # DQN and Target DQN
        model = get_models(args.model)
        self.dqn = model(actions=args.actions, atoms=args.atoms)
        self.target_dqn = model(actions=args.actions, atoms=args.atoms)

        dqn_params = 0
        for weight in self.dqn.parameters():
            weight_params = 1
            for s in weight.size():
                weight_params *= s
            dqn_params += weight_params
        print("Distrib DQN has {:,} parameters.".format(dqn_params))

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

    def sync_target_network(self):
        for target, source in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target.data = source.data

    def act(self, state, epsilon, exp_model, evaluation=False):
        # self.T += 1
        self.dqn.eval()
        orig_state = state[:, :, -1:]
        state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
        q_values_distributions = self.dqn(Variable(state, volatile=True)).cpu().data[0]
        # TODO: Log Q-Value distributions
        # print(q_values_distributions)
        values = torch.linspace(self.args.v_min, self.args.v_max, steps=self.args.atoms)
        # print(values)
        q_value_expectations = q_values_distributions @ values
        # print(q_value_expectations)
        q_values_numpy = q_value_expectations.numpy()

        extra_info = {}

        if self.args.optimistic_init and not evaluation:
            raise NotImplementedError
            q_values_pre_bonus = np.copy(q_values_numpy)
            if not self.args.ucb:
                for a in range(self.args.actions):
                    _, info = exp_model.bonus(orig_state, a, dont_remember=True)
                    action_pseudo_count = info["Pseudo_Count"]
                    # TODO: Log the optimism bonuses
                    optimism_bonus = self.args.optimistic_scaler / np.sqrt(action_pseudo_count + 0.01)
                    self.log("Bandit/Action_{}".format(a), optimism_bonus, step=self.T)
                    q_values[a] += optimism_bonus
            else:
                action_counts = []
                for a in range(self.args.actions):
                    _, info = exp_model.bonus(orig_state, a, dont_remember=True)
                    action_pseudo_count = info["Pseudo_Count"]
                    action_counts.append(action_pseudo_count)
                total_count = sum(action_counts)
                for ai, a in enumerate(action_counts):
                    # TODO: Log the optimism bonuses
                    optimisim_bonus = self.args.optimistic_scaler * np.sqrt(2 * np.log(max(1, total_count)) / (a + 0.01))
                    self.log("Bandit/UCB/Action_{}".format(ai), optimisim_bonus, step=self.T)
                    q_values[ai] += optimisim_bonus

            extra_info["Action_Bonus"] = q_values_numpy - q_values_pre_bonus

        extra_info["Q_Values"] = q_values_numpy

        if np.random.random() < epsilon:
            action = np.random.randint(low=0, high=self.args.actions)
        else:
            action = np.argmax(q_values_numpy)
            # action = q_values.max(0)[1][0]  # Torch...

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

            batch, indices, is_weights = self.replay.Sample_N(self.args.batch_size, self.args.n_step, self.args.gamma)
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

            # print(q_value_targets)
            q_value_targets = q_value_targets.view(self.args.batch_size, 1, 1)
            q_value_targets = q_value_targets.expand(self.args.batch_size, self.args.actions, self.args.atoms)
            # q_value_targets = q_value_targets.repeat(self.args.batch_size, self.args.actions, 1)
            # print(q_value_targets, target_dqn_qvals)
            q_value_targets = q_value_targets * target_dqn_qvals_data
            # print(q_value_targets)

            # print(q_value_targets, rewards)
            rewards = rewards.view(self.args.batch_size, 1, 1)
            rewards = rewards.expand(self.args.batch_size, self.args.actions, self.args.atoms)
            q_value_targets = q_value_targets + rewards
            # print(q_value_targets)

            # Have to project the q_value_targets back onto the original support

            # AM DOING THIS ALL WRONG, WE SHOULD BE USING THE ATOMS NOT THE PROBABILITIES

            self.dqn.train()
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

        # Pad out the states to be of size batch_size
        if len(info["States"]) < self.args.batch_size:
            old_states = info["States"]
            new_states = old_states[0] * (self.args.batch_size - len(old_states))
            info["States"] = new_states

        return info

    def train_elig(self):
        if self.T - self.target_sync_T > self.args.target:
            self.sync_target_network()
            self.target_sync_T = self.T

        self.dqn.eval()

        # TODO: Use a named tuple for experience replay
        experiences, batch = self.replay.Sample_N_Eligibility_States(self.args.batch_size, self.args.gamma, self.args.num_states, self.args.gap)
        columns = list(zip(*batch))

        states = Variable(torch.from_numpy(np.array(columns[0])).float().transpose_(1, 3))
        actions = Variable(torch.LongTensor(columns[1]))

        elig_states = []
        for seq in experiences:
            for triple in seq:
                elig_states.append(triple[0][0])
        new_states = Variable(torch.from_numpy(np.array(elig_states)).float().transpose_(1, 3))

        target_dqn_qvals = self.target_dqn(new_states).cpu()

        q_targets = []

        q_state_index = 0
        for seq in experiences:

            averaged_return = 0
            lambda_ = 1
            for triple in seq:
                exp = triple[0]
                accum_reward = triple[1]
                # Clip
                if accum_reward > 1:
                    accum_reward = 1
                if accum_reward < -1:
                    accum_reward = -1

                steps = triple[2]
                terminal = exp.terminal

                k_step_return = Variable(torch.FloatTensor([accum_reward] * self.args.actions))
                if not terminal:
                    k_step_return += (self.args.gamma ** steps) * target_dqn_qvals[q_state_index]
                q_state_index += 1
                averaged_return += lambda_ * k_step_return
                lambda_ *= self.args.lambda_
            # if len(seq) < 1:
            #     # print(experiences)
            #     # print(batch)
            #     continue
            averaged_return /= sum([self.args.lambda_ ** t for t in range(len(seq))])

            max_q_val = averaged_return.data.max(0)[0][0]
            # max_q_val = max_q_val.float()
            q_targets.append(max_q_val)

        q_value_targets = Variable(torch.from_numpy(np.array(q_targets)))
        q_value_targets = q_value_targets.float()
        # q_value_targets = q_targets

        self.dqn.train()
        if self.args.gpu:
            actions = actions.cuda()
            q_value_targets = q_value_targets.cuda()
        model_predictions = self.dqn(states).gather(1, actions.view(-1, 1))
        # print(q_value_targets)
        # print(model_predictions)

        info = {}

        td_error = model_predictions - q_value_targets
        info["TD_Error"] = td_error.mean().data[0]

        # Update the priorities
        # self.replay.Update_Indices(indices, td_error.cpu().data.numpy())

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

        return info
