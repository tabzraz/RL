import argparse
import gym
import datetime
import time
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Utils.Utils import time_str
from Replay.ExpReplay_Options import ExperienceReplay_Options
from Models.Models import get_models
import Envs

parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--t-max", type=int, default=int(1e5))
parser.add_argument("--env", type=str, default="Maze-2-v1")
parser.add_argument("--logdir", type=str, default="Logs")
parser.add_argument("--name", type=str, default="nn")
parser.add_argument("--exp-replay-size", "--xp", type=int, default=int(1e4))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--model", type=str, default="Maze-2")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--exploration-steps", "--exp-steps", type=int, default=int(1e5))
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--epsilon-steps", "--eps-steps", type=int, default=int(5e4))
parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
parser.add_argument("--epsilon-finish", "--eps-finish", type=float, default=0.1)
parser.add_argument("--batch-size", "--batch", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
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
dqn = get_models(args.model)()
target_dqn = get_models(args.model)()

if args.gpu:
    dqn.cuda()
    target_dqn.cuda()

# Optimizer
optimizer = optim.Adam(dqn.parameters(), lr=args.lr)

# Stuff to log
Q_Values = []
Episode_Rewards = []
Episode_Lengths = []
Rewards = []
DQN_Loss = []


# Variables and stuff
T = 1
episode = 1

start_time = time.time()


# Methods
def select_action(state):
    state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
    q_values = dqn(Variable(state, volatile=True)).data[0]
    Q_Values.append(q_values)
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = q_values.max(0)[1][0]  # Torch...
    return action


def explore():
    print("\nExploratory phase for {} steps".format(args.exploration_steps))
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
    time_elapsed = time.time() - start_time
    time_left = time_elapsed * (args.t_max - T) / T
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    print("\033[F\033[F\x1b[KEp: {:,}, T: {:,}/{:,}, \n\x1b[KEpsilon: {:.2f}, Elapsed: {}, Left: {}\n".format(episode, T, args.t_max, epsilon, time_str(time_elapsed), time_str(time_left)), " " * 10, end="\r")


def epsilon_schedule():
    return args.epsilon_finish + (args.epsilon_start - args.epsilon_finish) * max(((args.epsilon_steps - T) / args.epsilon_steps), 0)


def train_agent():
    # TODO: Use a named tuple for experience replay
    batch = replay.Sample(args.batch_size)
    columns = list(zip(*batch))

    states = Variable(torch.from_numpy(np.array(columns[0])).float().transpose_(1, 3))
    actions = Variable(torch.LongTensor(columns[1]))
    terminal_states = torch.FloatTensor(columns[5])
    rewards = torch.FloatTensor(columns[2])
    steps = torch.FloatTensor(columns[4])
    new_states = Variable(torch.from_numpy(np.array(columns[3])).float().transpose_(1, 3))

    target_dqn_qvals = target_dqn(new_states).data
    new_states_qvals = dqn(new_states).data

    q_value_targets = (torch.ones(args.batch_size) - terminal_states)
    q_value_targets *= torch.pow(torch.ones(args.batch_size) * args.gamma, steps)
    # Double Q Learning
    q_value_targets *= target_dqn_qvals.gather(1, new_states_qvals.max(1)[1])
    q_value_targets += rewards

    model_predictions = dqn(states).gather(1, actions.view(-1, 1))

    l2_loss = (model_predictions - Variable(q_value_targets)).pow(2).mean()
    DQN_Loss.append(l2_loss.data[0])

    # Update
    optimizer.zero_grad()
    l2_loss.backward()
    optimizer.step()


explore()

print("Training\n\n\n")

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
        episode_steps += 1
        T += 1
        Rewards.append(reward)

        replay.Add_Exp(state, action, reward, state_new, 1, episode_finished)

        train_agent()

        state = state_new

        print("\x1b[K" + "." * ((episode_steps // 20) % 40), end="\r")

    episode += 1
    Episode_Rewards.append(episode_reward)
    Episode_Lengths.append(episode_steps)
    Rewards = []
