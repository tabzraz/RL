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
        self.logger = logging_func["logger"]

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
        if self.T % self.args.tb_interval == 0 and not evaluation:
            for i in range(self.args.actions):
                self.logger.add_histogram("Q_Values_Distrib_Action_{}".format(i), q_values_distributions[i].numpy(), self.T)
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
            action = int(np.argmax(q_values_numpy))
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

            q_value_gammas = (Variable(torch.ones(terminal_states.size()[0])) - terminal_states)
            inter = Variable(torch.ones(terminal_states.size()[0]) * self.args.gamma)
            # print(steps)
            q_value_gammas = q_value_gammas * torch.pow(inter, steps)

            values = torch.linspace(self.args.v_min, self.args.v_max, steps=self.args.atoms)
            values = Variable(values)
            values = values.view(1, 1, self.args.atoms)
            values = values.expand(self.args.batch_size, self.args.actions, self.args.atoms)
            # print(values)

            q_value_gammas = q_value_gammas.view(self.args.batch_size, 1, 1)
            q_value_gammas = q_value_gammas.expand(self.args.batch_size, self.args.actions, self.args.atoms)
            # print(q_value_gammas)
            gamma_values = q_value_gammas * values
            # print(gamma_values)
            rewards = rewards.view(self.args.batch_size, 1, 1)
            rewards = rewards.expand(self.args.batch_size, self.args.actions, self.args.atoms)
            # print(rewards)
            operator_q_values = rewards + gamma_values
            # print(operator_q_values)

            clipped_operator_q_values = torch.clamp(operator_q_values, self.args.v_min, self.args.v_max)

            delta_z = (self.args.v_max - self.args.v_min) / (self.args.atoms - 1)
            # Using the notation from the categorical paper
            b_j = (clipped_operator_q_values - self.args.v_min) / delta_z
            # print(b_j)
            lower_bounds = torch.floor(b_j)
            upper_bounds = torch.ceil(b_j)

            # Work out the max action
            atom_values = Variable(torch.linspace(self.args.v_min, self.args.v_max, steps=self.args.atoms))
            atom_values = atom_values.view(1, 1, self.args.atoms)
            atom_values = atom_values.expand(self.args.batch_size, self.args.actions, self.args.atoms)

            # Sum over the atoms dimension
            target_expected_qvalues = torch.sum(target_dqn_qvals_data * atom_values, dim=2)
            # Get the maximum actions index across the batch size
            max_actions = target_expected_qvalues.max(dim=1)[1].view(-1)

            # Project back onto the original support for the max actions
            q_value_distribution_targets = torch.zeros(self.args.batch_size, self.args.atoms)

            # Distributions for the max actions
            # print(target_dqn_qvals_data, max_actions)
            q_value_max_actions_distribs = target_dqn_qvals_data.index_select(dim=1, index=max_actions)[:,0,:]
            # print(q_value_max_actions_distribs)

            # Lower_bounds_actions
            lower_bounds_actions = lower_bounds.index_select(dim=1, index=max_actions)[:,0,:]
            upper_bounds_actions = upper_bounds.index_select(dim=1, index=max_actions)[:,0,:]
            b_j_actions = b_j.index_select(dim=1, index=max_actions)[:,0,:]

            lower_bound_values_to_add = q_value_max_actions_distribs * (upper_bounds_actions - b_j_actions)
            upper_bound_values_to_add = q_value_max_actions_distribs * (b_j_actions - lower_bounds_actions)
            # print(lower_bounds_actions)
            # print(lower_bound_values_to_add)
            # Naive looping
            for b in range(self.args.batch_size):
                for l, pj in zip(lower_bounds_actions.data.type(torch.LongTensor)[b], lower_bound_values_to_add[b].data):
                    q_value_distribution_targets[b][l] += pj
                for u, pj in zip(upper_bounds_actions.data.type(torch.LongTensor)[b], upper_bound_values_to_add[b].data):
                    q_value_distribution_targets[b][u] += pj

            self.dqn.train()
            if self.args.gpu:
                actions = actions.cuda()
                q_value_targets = q_value_targets.cuda()
            model_predictions = self.dqn(states).index_select(1, actions.view(-1))[:,0,:]
            q_value_distribution_targets = Variable(q_value_distribution_targets)
            # print(q_value_distribution_targets)
            # print(model_predictions) 

            # Cross entropy loss
            ce_loss = -torch.sum(q_value_distribution_targets * torch.log(model_predictions), dim=1)
            ce_batch_loss = ce_loss.mean()

            info = {}

            self.log("DQN/X_Entropy_Loss", ce_batch_loss.data[0], step=self.T)

            # Update
            self.optimizer.zero_grad()
            ce_batch_loss.backward()

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
