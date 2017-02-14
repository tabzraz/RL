import argparse
import gym
import datetime
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Utils.Utils import time_str
from Replay.ExpReplay_Options import ExperienceReplay_Options
from Models import Models
import Envs

parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--t-max", type=int, default=int(1e6))
parser.add_argument("--env", type=str, default="Maze-2-v1")
parser.add_argument("--logdir", type=str, default="Logs")
parser.add_argument("--name", type=str, default="nn")
parser.add_argument("--xp", type=int, default=int(1e4))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-gpu", action="store_true", default=False)
parser.add_argument("--model", type=str, default="Maze-2")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--exp-steps", "--exploration-steps", type=int, default=1e5)
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--epsilon-steps", "--eps-steps", type=int, default=int(50e5))
parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
parser.add_argument("--epsilon-finish", "--eps-finish", type=float, default=0.1)
parser.add_argument("--batch-size", "--batch", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
args = parser.parse_args()
args.gpu = not args.no_gpu and torch.cuda.is_available()
print("Settings:\n", args, "\n")

LOGDIR = "{}/{}_{}".format(args.logdir, args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

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
replay = ExperienceReplay_Options(args.xp_size)

# DQN
dqn = Models[args.model]()
target_dqn = Models[args.model]()

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
    state = torch.from_numpy(state).float().unsqueeze(0)
    q_values = dqn(Variable(state, volatile=True))[0].cpu()
    Q_Values.append(q_values)
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_values)
    return action


def explore():
    print("Exploratory phase for {} steps".format(args.exploration_steps))
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

    print("Exploratory phase finished, starting learning")


def print_time():
    time_elapsed = time.time() - start_time
    time_left = time_elapsed * (args.t_max - T) / T
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    print("Ep: {:,}, T: {:,}/{:,}, Epsilon: {:.2f}, Elapsed: {}, Left: {}".format(episode, T, args.t_max, epsilon, time_str(time_elapsed), time_str(time_left)), " " * 10, end="\r")


def epsilon_schedule():
    return args.epsilon_finish + (args.epsilon_start - args.epsilon_finish) * max(((args.epsilon_steps - T) / args.epsilon_steps), 0)


def train_agent():
    # TODO: Use a named tuple for experience replay
    batch = replay.sample(args.batch_size)
    columns = list(zip(*batch))

    states = Variable(columns[0])
    actions = torch.IntTensor(columns[1])
    terminal_states = torch.FloatTensor(columns[5])
    rewards = torch.FloatTensor(columns[2])
    steps = torch.IntTensor(columns[4])
    new_states = Variable(columns[3], volatile=True)

    target_dqn_qvals = target_dqn(new_states)
    new_states_qvals = dqn(new_states)

    q_value_targets = (torch.ones(args.batch_size) - terminal_states)
    q_value_targets *= torch.pow(torch.ones(args.batch_size) * args.gamma, steps)
    # Double Q Learning
    q_value_targets *= target_dqn_qvals[np.argmax(new_states_qvals, axis=1)]
    q_value_targets += rewards

    model_predictions = dqn(states).gather(1, actions)

    l2_loss = (model_predictions - q_value_targets).pow(2).mean()
    DQN_Loss.append(l2_loss.data[0])

    # Update
    optimizer.zero_grad()
    l2_loss.backward()
    optimizer.step()


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

    episode += 1
    Episode_Rewards.append(episode_reward)
    Episode_Lengths.append(episode_steps)
    Rewards = []



# FLAGS = flags.FLAGS
# ENV_NAME = FLAGS.env
# RENDER = FLAGS.render
# SLOW = FLAGS.slow
# env = gym.make(ENV_NAME)

# if FLAGS.action_override > 0:
#     ACTIONS = FLAGS.action_override
# else:
#     ACTIONS = env.action_space.n
# DOUBLE_DQN = FLAGS.double
# SEED = FLAGS.seed
# LR = FLAGS.lr
# VIME_LR = FLAGS.vime_lr
# ETA = FLAGS.eta
# VIME_BATCH_SIZE = FLAGS.vime_batch
# VIME_ITERS = FLAGS.vime_iters
# NAME = FLAGS.name
# EPISODES = FLAGS.episodes
# T_MAX = FLAGS.t_max
# EPSILON_START = FLAGS.epsilon_start
# EPSILON_FINISH = FLAGS.epsilon_finish
# EPSILON_STEPS = FLAGS.epsilon_steps
# XP_SIZE = FLAGS.xp
# GAMMA = FLAGS.gamma
# BATCH_SIZE = FLAGS.batch
# TARGET_UPDATE = FLAGS.target
# SUMMARY_UPDATE = FLAGS.summary
# CLIP_VALUE = FLAGS.grad_clip
# VIME = FLAGS.vime
# VIME_POSTERIOR_ITERS = FLAGS.posterior_iters
# RANDOM_AGENT = FLAGS.rnd
# CHECKPOINT_INTERVAL = FLAGS.ckpt_interval
# if FLAGS.ckpt != "":
#     RESTORE = True
#     CHECKPOINT = FLAGS.ckpt
# else:
#     RESTORE = False
# EXPLORATION_STEPS = FLAGS.exp_steps
# if FLAGS.logdir != "":
#     LOGDIR = FLAGS.logdir
# else:
#     LOGDIR = "Logs/{}_{}".format(NAME, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
# DQN_MODEL = FLAGS.model
# LEVELS = FLAGS.levels

# if not os.path.exists("{}/ckpts/".format(LOGDIR)):
#     os.makedirs("{}/ckpts".format(LOGDIR))

# # Print all the hyperparams
# hyperparams = FLAGS.__dict__["__flags"]
# with open("{}/info.json".format(LOGDIR), "w") as fp:
#     json.dump(hyperparams, fp)

# # TODO: Add some more info here
# print("\n--------Info--------")
# print("Logdir:", LOGDIR)
# print("T: {:,}".format(T_MAX))
# print("Actions:", ACTIONS)
# print("Gamma", GAMMA)
# print("Learning Rate:", LR)
# print("Batch Size:", BATCH_SIZE)
# print("VIME bonuses:", VIME)
# print("--------------------\n")

# replays = [ExperienceReplay(XP_SIZE) for _ in range(LEVELS)]

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.Graph().as_default():
#     with tf.Session(config=config) as sess:

#         # Seed numpy and tensorflow
#         np.random.seed(SEED)
#         tf.set_random_seed(SEED)

#         print("\n-------Models-------")

#         dqn_creator = get_models(DQN_MODEL)
#         dqn = dqn_creator("DQN")
#         target_dqn = dqn_creator("Target_DQN")
