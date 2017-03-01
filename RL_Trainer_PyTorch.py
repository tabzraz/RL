import argparse
import gym
import datetime
import time
import os
from math import sqrt

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

import imageio
# from pygame.image import tostring as pygame_tostring
from pygame.surfarray import array3d as pygame_image
# import torch.nn.modules.utils.clip_grad_norm as clip_grad

import Exploration.CTS as CTS

# from pycrayon import CrayonClient
from tensorboard_logger import configure
from tensorboard_logger import log_value as tb_log_value
from multiprocessing import Process, Queue


from Utils.Utils import time_str
from Replay.ExpReplay_Options import ExperienceReplay_Options
from Models.Models import get_torch_models as get_models
import Envs

parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--t-max", type=int, default=int(2e5))
parser.add_argument("--env", type=str, default="Maze-2-v1")
parser.add_argument("--logdir", type=str, default="Logs")
parser.add_argument("--name", type=str, default="nn")
parser.add_argument("--exp-replay-size", "--xp", type=int, default=int(2e4))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--model", type=str, default="")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--exploration-steps", "--exp-steps", type=int, default=int(1e4))
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--epsilon-steps", "--eps-steps", type=int, default=int(1e5))
parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
parser.add_argument("--epsilon-finish", "--eps-finish", type=float, default=0.1)
parser.add_argument("--batch-size", "--batch", type=int, default=32)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target", type=int, default=100)
parser.add_argument("--count", action="store_true", default=False)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--n-step", "--n", type=int, default=1)
parser.add_argument("--plain-print", action="store_true", default=False)
parser.add_argument("--clip-value", type=float, default=5)
parser.add_argument("--no-tb", action="store_true", default=False)
parser.add_argument("--no-eval-images", action="store_true", default=False)
parser.add_argument("--eval-images-interval", type=int, default=25)
parser.add_argument("--debug-eval", action="store_true", default=False)
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

NAME_DATE = "{}_{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
LOGDIR = "{}/{}".format(args.logdir, NAME_DATE)
print("Logging to:\n{}\n".format(LOGDIR))

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists("{}/logs".format(LOGDIR)):
    os.makedirs("{}/logs".format(LOGDIR))
if not os.path.exists("{}/evals".format(LOGDIR)):
    os.makedirs("{}/evals".format(LOGDIR))

with open("{}/settings.txt".format(LOGDIR), "w") as f:
    f.write(str(args))

# Seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu:
    torch.cuda.manual_seed_all(args.seed)

# TB
args.tb = not args.no_tb
# Saving the evaluation policies as images
args.eval_images = not args.no_eval_images
# Model
if args.model == "":
    args.model = args.env

# Gym Environment
env = gym.make(args.env)
args.actions = env.action_space.n

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
Episode_Bonus_Only_Rewards = []
Episode_Lengths = []
Rewards = []
States = []
Actions = []
States_Next = []
DQN_Loss = []
DQN_Grad_Norm = []
Exploration_Bonus = []

Last_T_Logged = 1
Last_Ep_Logged = 1


# Variables and stuff
T = 1
episode = 1
epsilon = 1
episode_reward = 0
episode_bonus_only_reward = 0
epsiode_steps = 0

# Debug stuff
max_q_value = -1000
max_exp_bonus = 0

# Async queue
log_queue = Queue()
eval_images_queue = Queue()
eval_images = 0

if args.count:
    cts_model = CTS.LocationDependentDensityModel(frame_shape=(env.shape[0] * 7, env.shape[0] * 7, 1), context_functor=CTS.L_shaped_context)


# class Variable(torch.autograd.Variable):

#     def __init__(self, data, *arguments, **kwargs):
#         if args.gpu:
#             data = data.cuda()
#             print(data)
#         super(Variable, self).__init__(data, *arguments, **kwargs)

# Multiprocessing logger
def logger(q):
    configure("{}/tb".format(LOGDIR), flush_secs=30)
    # Crayon stuff
    # crayon_client = CrayonClient(hostname="localhost")
    # crayon_exp = crayon_client.create_experiment(NAME_DATE)
    while True:
        (name, value, step) = q.get(block=True)
        tb_log_value(name, value, step=step)
        # crayon_exp.add_scalar_value(name, value, step=step)


def log_value(name, value, step):
    log_queue.put((name, value, step))


# Methods
def environment_specific_stuff():
    if args.env.startswith("Maze"):
        player_pos = env.player_pos
        with open("{}/logs/Player_Positions_In_Maze.txt".format(LOGDIR), "a") as file:
            file.write(str(player_pos) + "\n")


def eval_agent(last=False):
    # Best to keep this on the main thread so we have access to the CTS model
    # If this becomes a bottleneck then think about asyncing this as a whole
    global epsilon
    epsilon = 0
    terminated = False
    ep_reward = 0
    steps = 0
    state = env.reset()
    states = [state]
    Eval_Q_Values = []

    global eval_images
    will_save_states = args.eval_images and (last or eval_images % args.eval_images_interval == 0)

    if will_save_states and args.debug_eval:
        env.debug_render(offline=True)
        debug_states = [pygame_image(env.surface)]

    while not terminated:
        action, q_vals = select_action(state, training=False)
        state_new, reward, terminated, env_info = env.step(action)
        ep_reward += reward
        steps += 1

        Eval_Q_Values.append(q_vals)

        if will_save_states:
            # Only do this stuff if we're actually gonna save it
            if args.debug_eval:
                debug_dict = {}
                debug_dict["Q_Values"] = q_vals
                debug_dict["Max_Q_Value"] = max_q_value
                debug_dict["Action"] = action
                if args.count:
                    exp_bonus = exploration_bonus(state, training=False)
                    debug_dict["Max_Exp_Bonus"] = max_exp_bonus
                    debug_dict["Exp_Bonus"] = exp_bonus
                env.debug_render(debug_info=debug_dict, offline=True)
                debug_states.append(pygame_image(env.surface))
            states.append(state)

        state = state_new

    # Log the Q_Values for this
    with open("{}/logs/Eval_Q_Values_T.txt".format(LOGDIR), "ab") as file:
        np.savetxt(file, Eval_Q_Values[:], delimiter=" ", fmt="%f")
        file.write(str.encode("\n"))

    if args.tb:
        log_value("Eval/Episode_Reward", ep_reward, step=T)
        log_value("Eval/Episode_Length", steps, step=T)

    if will_save_states:
        if args.debug_eval:
            save_states(debug_states, debug=True)
        else:
            save_states(states)
    eval_images += 1


def save_states(states, debug=False):
    eval_images_queue.put((states, debug, LOGDIR, T, episode))


def gif_saver(q):
    while True:
        (states, debug, LOGDIR, T, episode) = q.get(block=True)
        images = []
        # Pray this keeps working
        for s in states:
            if debug:
                images.append(s.swapaxes(0, 1))
            else:
                images.append(s[:, :, 0] * 255)
        imageio.mimsave("{}/evals/T_{}__Ep_{}.gif".format(LOGDIR, T, episode), images)


def exploration_bonus(state, training=True):
    bonus = 0
    if args.count:
        rho_old = np.exp(cts_model.update(state))
        rho_new = np.exp(cts_model.log_prob(state))
        pseudo_count = (rho_old * (1 - rho_new)) / (rho_new - rho_old)
        pseudo_count = max(pseudo_count, 0)
        bonus = args.beta / sqrt(pseudo_count + 0.01)
        if training:
            Exploration_Bonus.append(bonus)
            if args.tb:
                log_value("Count_Bonus", bonus, step=T)
    global max_exp_bonus
    max_exp_bonus = max(max_exp_bonus, bonus)
    return bonus


def start_of_episode():
    if args.tb:
        log_value("Epsilon", epsilon, step=T)


def end_of_episode():
    if args.tb:
        log_value("Episode_Reward", episode_reward, step=T)
        log_value("Episode_Bonus_Only_Reward", episode_bonus_only_reward, step=T)
        log_value("Episode_Length", episode_steps, step=T)


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


def select_action(state, training=True):
    state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
    q_values = dqn(Variable(state, volatile=True)).cpu().data[0]
    q_values_numpy = q_values.numpy()

    global max_q_value
    max_q_value = max(max_q_value, np.max(q_values_numpy))

    if training:
        Q_Values.append(q_values_numpy)

        # Log the q values
        if args.tb:
            # crayon_exp.add_histogram_value("DQN/Q_Values", q_values_numpy.tolist(), tobuild=True, step=T)
            # q_val_dict = {}
            for index in range(args.actions):
                # q_val_dict["DQN/Action_{}_Q_Value".format(index)] = float(q_values_numpy[index])
                log_value("DQN/Action_{}_Q_Value".format(index), q_values_numpy[index], step=T)
            # print(q_val_dict)
            # crayon_exp.add_scalar_dict(q_val_dict, step=T)

    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = q_values.max(0)[1][0]  # Torch...

    return action, q_values_numpy


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

    target_dqn_qvals = target_dqn(new_states).cpu()
    new_states_qvals = dqn(new_states).cpu()
    # Make a new variable with those values so that these are treated as constants
    target_dqn_qvals_data = Variable(target_dqn_qvals.data)
    new_states_qvals_data = Variable(new_states_qvals.data)

    q_value_targets = (Variable(torch.ones(args.batch_size)) - terminal_states)
    inter = Variable(torch.ones(args.batch_size) * args.gamma)
    # print(steps)
    q_value_targets = q_value_targets * torch.pow(inter, steps)
    # Double Q Learning
    q_value_targets = q_value_targets * target_dqn_qvals_data.gather(1, new_states_qvals_data.max(1)[1])
    q_value_targets = q_value_targets + rewards

    if args.gpu:
        actions = actions.cuda()
        q_value_targets = q_value_targets.cuda()
    model_predictions = dqn(states).gather(1, actions.view(-1, 1))

    td_error = model_predictions - q_value_targets
    l2_loss = (td_error).pow(2).mean()
    DQN_Loss.append(l2_loss.data[0])

    # Update
    optimizer.zero_grad()
    l2_loss.backward()

    # Taken from pytorch clip_grad_norm
    # Remove once the pip version it up to date with source
    total_norm = 0
    for p in dqn.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    DQN_Grad_Norm.append(total_norm)
    clip_coef = float(args.clip_value) / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in dqn.parameters():
            p.grad.data.mul_(clip_coef)

    optimizer.step()

    # Crayon
    if args.tb:
        log_value("DQN/Gradient_Norm", total_norm, step=T)
        log_value("DQN/Loss", l2_loss.data[0], step=T)
        log_value("DQN/TD_Error", td_error.mean().data[0], step=T)


######################
# Training procedure #
######################

# Start the async logger
p_log = Process(target=logger, args=(log_queue,), daemon=True)
p_gif = Process(target=gif_saver, args=(eval_images_queue,), daemon=True)
p_log.start()
p_gif.start()

explore()

sync_target_network()

start_time = time.time()

print("Training.\n\n\n")

while T < args.t_max:

    state = env.reset()
    if args.render:
        env.debug_render()
    episode_finished = False
    episode_reward = 0
    episode_bonus_only_reward = 0
    episode_steps = 0

    epsilon = epsilon_schedule()

    start_of_episode()

    print_time()

    while not episode_finished:
        action, q_values = select_action(state)
        state_new, reward, episode_finished, env_info = env.step(action)
        episode_steps += 1
        T += 1
        # If the environment terminated because it reached a limit, we do not want the agent
        # to see that transition, since it makes the env non markovian wrt state
        if "Steps_Termination" in env_info:
            episode_finished = True
            break

        episode_reward += reward

        exp_bonus = exploration_bonus(state)
        episode_bonus_only_reward += exp_bonus
        reward += exp_bonus

        if args.render:
            debug_dict = {}
            debug_dict["Q_Values"] = q_values
            debug_dict["Max_Q_Value"] = max_q_value
            debug_dict["Action"] = action
            if args.count:
                debug_dict["Max_Exp_Bonus"] = max_exp_bonus
                debug_dict["Exp_Bonus"] = exp_bonus
            env.debug_render(debug_dict)

        Rewards.append(reward)
        States.append(state)
        States_Next.append(state_new)
        Actions.append(action)

        replay.Add_Exp(state, action, reward, state_new, 1, episode_finished)

        train_agent()

        if T % args.target == 0:
            sync_target_network()

        state = state_new

        environment_specific_stuff()

        if not args.plain_print:
            print("\x1b[K" + "." * ((episode_steps // 20) % 40), end="\r")

    eval_agent()

    episode += 1
    Episode_Rewards.append(episode_reward)
    Episode_Lengths.append(episode_steps)
    Episode_Bonus_Only_Rewards.append(episode_bonus_only_reward)

    Rewards = []
    States = []
    States_Next = []
    Actions = []

    end_of_episode()

    save_values()


eval_agent(last=True)

q.close()
p.terminate()
print("\nFinished\n")
