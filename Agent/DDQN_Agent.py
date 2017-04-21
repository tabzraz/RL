import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from Replay.ExpReplay_Options import ExperienceReplay_Options


class DDQN_Agent:

    def __init__(self, model, args):
        self.args = args

        # Experience Replay
        self.replay = ExperienceReplay_Options(args.exp_replay_size)

        # DQN and Target DQN
        self.dqn = model(actions=args.actions)
        self.target_dqn = model(actions=args.actions)

        dqn_params = 0
        for weight in self.dqn.parameters():
            weight_params = 1
            for s in weight.size():
                weight_params *= s
            dqn_params += weight_params
        print("\nDQN has {:,} parameters.".format(dqn_params))

        self.target_dqn.eval()

        if args.gpu:
            print("Moving models to GPU.")
            self.dqn.cuda()
            self.target_dqn.cuda()

        # Optimizer
        self.optimizer = Adam(self.dqn.parameters(), lr=args.lr)

        self.T = 0
        self.target_sync_T = -self.args.t_max

    def sync_target_network(self):
        for target, source in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target.data = source.data

    def act(self, state, epsilon, exp_model):
        self.T += 1
        self.dqn.eval()
        state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
        q_values = self.dqn(Variable(state, volatile=True)).cpu().data[0]
        q_values_numpy = q_values.numpy()

        extra_info = {}
        extra_info["Q_Values"] = q_values_numpy

        if np.random.random() < epsilon:
            action = np.random.randint(low=0, high=self.args.actions)
        else:
            action = q_values.max(0)[1][0]  # Torch...

        extra_info["Action"] = action

        return action, extra_info

    def experience(self, state, action, reward, state_next, steps, terminated):
        self.replay.Add_Exp(state, action, reward, state_next, steps, terminated)

    def train(self):
        if self.T - self.target_sync_T > self.args.target:
            self.sync_target_network()
            self.target_sync_T = self.T

        self.dqn.eval()

        # TODO: Use a named tuple for experience replay
        batch = self.replay.Sample_N(self.args.batch_size, self.args.n_step, self.args.gamma)
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
        new_states_qvals = self.dqn(new_states).cpu()
        # Make a new variable with those values so that these are treated as constants
        target_dqn_qvals_data = Variable(target_dqn_qvals.data)
        new_states_qvals_data = Variable(new_states_qvals.data)

        q_value_targets = (Variable(torch.ones(terminal_states.size()[0])) - terminal_states)
        inter = Variable(torch.ones(terminal_states.size()[0]) * self.args.gamma)
        # print(steps)
        q_value_targets = q_value_targets * torch.pow(inter, steps)
        # Double Q Learning
        q_value_targets = q_value_targets * target_dqn_qvals_data.gather(1, new_states_qvals_data.max(1)[1])
        q_value_targets = q_value_targets + rewards

        self.dqn.train()
        if self.args.gpu:
            actions = actions.cuda()
            q_value_targets = q_value_targets.cuda()
        model_predictions = self.dqn(states).gather(1, actions.view(-1, 1))

        info = {}

        td_error = model_predictions - q_value_targets
        info["TD_Error"] = td_error.mean().data[0]

        l2_loss = (td_error).pow(2).mean()
        info["Loss"] = l2_loss.data[0]

        # Update
        self.optimizer.zero_grad()
        l2_loss.backward()

        # Taken from pytorch clip_grad_norm
        # Remove once the pip version it up to date with source
        gradient_norm = clip_grad_norm(self.dqn.parameters(), self.args.clip_value)
        info["Norm"] = gradient_norm

        self.optimizer.step()

        return info
