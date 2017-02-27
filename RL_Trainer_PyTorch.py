import argparse
import gym
import datetime
import time
import os
from math import sqrt

import numpy as np

import torch
import torch.optim as optim

import Exploration.CTS as CTS

from Utils.Utils import time_str
from Replay.ExpReplay_Options import ExperienceReplay_Options
from Models.Models import get_torch_models
import Envs

parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--t-max", type=int, default=int(2e5))
parser.add_argument("--env", type=str, default="Maze-1-v1")
parser.add_argument("--logdir", type=str, default="Logs")
parser.add_argument("--name", type=str, default="nn")
parser.add_argument("--exp-replay-size", "--xp", type=int, default=int(1e4))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--model", type=str, default="Maze-2")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--exploration-steps", "--exp-steps", type=int, default=int(1e5))
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--epsilon-steps", "--eps-steps", type=int, default=int(1e5))
parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
parser.add_argument("--epsilon-finish", "--eps-finish", type=float, default=0.1)
parser.add_argument("--batch-size", "--batch", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target", type=int, default=100)
parser.add_argument("--count", action="store_true", default=False)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--n-step", "--n", type=int, default=1)
parser.add_argument("--plain-print", action="store_true", default=False)
args = parser.parse_args()
if args.gpu and not torch.cuda.is_available():
    print("CUDA unavailable! Switching to cpu only")
# print("Settings:\n", args, "\n")
print("\n" + "=" * 40)
print(15 * " " + "Settings:" + " " * 15)
print("=" * 40)
for arg in vars(args):
    print(" {}: {}".format(arg, getattr(args, arg)))
print("=" * 40)
print()

LOGDIR = "{}/{}_{}".format(args.logdir, args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
print("Logging to:\n{}\n".format(LOGDIR))

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists("{}/logs".format(LOGDIR)):
    os.makedirs("{}/logs".format(LOGDIR))

with open("{}/settings.txt".format(LOGDIR), "w") as f:
    f.write(str(args))

# Seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu:
    torch.cuda.manual_seed_all(args.seed)

# Gym Environment
env = gym.make(args.env)

# Experience Replay
replay = ExperienceReplay_Options(args.exp_replay_size)

# DQN
print("\nGetting Models.\n")
dqn = get_models(args.model)()
target_dqn = get_models(args.model)()

if args.gpu:
    print("Moving models to GPU.")
    dqn.cuda()
    target_dqn.cuda()

# Optimizer
optimizer = optim.Adam(dqn.parameters(), lr=args.lr)

# Stuff to log
Q_Values = []
Episode_Rewards = []
Episode_Lengths = []
Rewards = []
States = []
Actions = []
States_Next = []
DQN_Loss = []
Exploration_Bonus = []

Last_T_Logged = 1
Last_Ep_Logged = 1


# Variables and stuff
T = 1
episode = 1

if args.count:
    cts_model = CTS.LocationDependentDensityModel(frame_shape=(env.shape[0] * 7, env.shape[0] * 7, 1), context_functor=CTS.L_shaped_context)


class Variable(torch.autograd.Variable):

    def __init__(self, data, *arguments, **kwargs):
        if args.gpu:
            data = data.cuda()
        super(Variable, self).__init__(data, *arguments, **kwargs)


# Methods
def add_exploration_bonus(state):
    if args.count:
        rho_old = np.exp(cts_model.update(state))
        rho_new = np.exp(cts_model.log_prob(state))
        pseudo_count = (rho_old * (1 - rho_new)) / (rho_new - rho_old)
        pseudo_count = max(pseudo_count, 0)
        bonus = args.beta / sqrt(pseudo_count + 0.01)
        Exploration_Bonus.append(bonus)
        return bonus
    return 0


def save_values():
    global Last_Ep_Logged
    if episode > Last_Ep_Logged:
        with open("{}/logs/Episode_Rewards.txt".format(LOGDIR), "ab") as file:
            np.savetxt(file, Episode_Rewards[Last_Ep_Logged - 1:], delimiter=" ", fmt="%f")

        with open("{}/logs/Episode_Lengths.txt".format(LOGDIR), "ab") as file:
            np.savetxt(file, Episode_Lengths[Last_Ep_Logged - 1:], delimiter=" ", fmt="%d")

        Last_Ep_Logged = episode

    global Last_T_Logged
    if T > Last_T_Logged:
        with open("{}/logs/Q_Values_T.txt".format(LOGDIR), "ab") as file:
            np.savetxt(file, Q_Values[Last_T_Logged - 1:], delimiter=" ", fmt="%f")
            file.write(str.encode("\n"))

        with open("{}/logs/DQN_Loss_T.txt".format(LOGDIR), "ab") as file:
            np.savetxt(file, DQN_Loss[Last_T_Logged - 1:], delimiter=" ", fmt="%.10f")
            file.write(str.encode("\n"))

        if args.count:
            with open("{}/logs/Exploration_Bonus_T.txt".format(LOGDIR), "ab") as file:
                np.savetxt(file, Exploration_Bonus[Last_T_Logged - 1:], delimiter=" ", fmt="%.10f")
                file.write(str.encode("\n"))

        Last_T_Logged = T


def sync_target_network():
    for target, source in zip(target_dqn.parameters(), dqn.parameters()):
        target.data = source.data


def select_action(state):
    state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
    q_values = dqn(Variable(state, volatile=True)).cpu().data[0]
    Q_Values.append(q_values.numpy())
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = q_values.max(0)[1][0]  # Torch...
    return action


def explore():
    print("\nExploratory phase for {} steps.".format(args.exploration_steps))
    e_steps = 0
    while e_steps < args.exploration_steps:
        s = env.reset()
        terminated = False
        while not terminated:
            print(e_steps, end="\r")
            a = env.action_space.sample()
            sn, r, terminated, _ = env.step(a)
            replay.Add_Exp(s, a, r, sn, 1, terminated)
            s = sn
            e_steps += 1

    print("Exploratory phase finished. Starting learning.\n")


def print_time():
    if args.plain_print:
        print(T, end="\r")
    else:
        time_elapsed = time.time() - start_time
        time_left = time_elapsed * (args.t_max - T) / T
        # Just in case its over 100 days
        time_left = min(time_left, 60 * 60 * 24 * 100)
        last_reward = "N\A"
        if len(Episode_Rewards) > 10:
            last_reward = "{:.2f}".format(np.mean(Episode_Rewards[-10:-1]))
        print("\033[F\033[F\x1b[KEp: {:,}, T: {:,}/{:,}, Epsilon: {:.2f}, Reward: {}, \n\x1b[KElapsed: {}, Left: {}\n".format(episode, T, args.t_max, epsilon, last_reward, time_str(time_elapsed), time_str(time_left)), " " * 10, end="\r")


def epsilon_schedule():
    return args.epsilon_finish + (args.epsilon_start - args.epsilon_finish) * max(((args.epsilon_steps - T) / args.epsilon_steps), 0)


def train_agent():
    # TODO: Use a named tuple for experience replay
    batch = replay.Sample_N(args.batch_size, args.n_step, args.gamma)
    columns = list(zip(*batch))

    states = Variable(torch.from_numpy(np.array(columns[0])).float().transpose_(1, 3))
    actions = Variable(torch.LongTensor(columns[1]))
    terminal_states = Variable(torch.FloatTensor(columns[5]))
    rewards = Variable(torch.FloatTensor(columns[2]))
    steps = Variable(torch.FloatTensor(columns[4]))
    new_states = Variable(torch.from_numpy(np.array(columns[3])).float().transpose_(1, 3))

    target_dqn_qvals = target_dqn(new_states)
    new_states_qvals = dqn(new_states)
    # Make a new variable with those values so that these are treated as constants
    target_dqn_qvals_data = Variable(target_dqn_qvals.data)
    new_states_qvals_data = Variable(new_states_qvals.data)

    q_value_targets = (Variable(torch.ones(args.batch_size)) - terminal_states)
    inter = Variable(torch.ones(args.batch_size) * args.gamma)
    # print(steps)
    q_value_targets_ = q_value_targets * torch.pow(inter, steps)
    # Double Q Learning
    q_value_targets__ = q_value_targets_ * target_dqn_qvals_data.gather(1, new_states_qvals_data.max(1)[1])
    q_value_targets___ = q_value_targets__ + rewards

    model_predictions = dqn(states).gather(1, actions.view(-1, 1))

    l2_loss = (model_predictions - q_value_targets___).pow(2).mean()
    DQN_Loss.append(l2_loss.data[0])

    # Update
    optimizer.zero_grad()
    l2_loss.backward()
    optimizer.step()


######################
# Training procedure #
######################

explore()

start_time = time.time()

print("Training.\n\n\n")

while T < args.t_max:

    state = env.reset()
    if args.render:
        env.render()
    episode_finished = False
    episode_reward = 0
    episode_steps = 0

    epsilon = epsilon_schedule()

    print_time()

    while not episode_finished:
        action = select_action(state)
        state_new, reward, episode_finished, env_info = env.step(action)
        # If the environment terminated because it reached a limit, we do not want the agent
        # to see that transition, since it makes the env non markovian wrt state
        if "Steps_Termination" in env_info:
            episode_finished = True
            break
        if args.render:
            env.render()

        episode_reward += reward

        reward += add_exploration_bonus(state)

        episode_steps += 1
        T += 1

        Rewards.append(reward)
        States.append(state)
        States_Next.append(state_new)
        Actions.append(action)

        replay.Add_Exp(state, action, reward, state_new, 1, episode_finished)

        train_agent()

        if T % args.target == 0:
            sync_target_network()

        state = state_new

        if not args.plain_print:
            print("\x1b[K" + "." * ((episode_steps // 20) % 40), end="\r")

    episode += 1
    Episode_Rewards.append(episode_reward)
    Episode_Lengths.append(episode_steps)

    Rewards = []
    States = []
    States_Next = []
    Actions = []

    save_values()
