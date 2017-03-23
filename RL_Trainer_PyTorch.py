import argparse
import gym
import datetime
import time
import os
import pickle
from math import sqrt, ceil

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

import imageio
# from pygame.image import tostring as pygame_tostring
from pygame.surfarray import array3d as pygame_image
# import torch.nn.modules.utils.clip_grad_norm as clip_grad

import Exploration.CTS as CTS
# from skimage.transform import resize
from scipy.misc import imresize as resize

# from pycrayon import CrayonClient
from tensorboard_logger import configure
from tensorboard_logger import log_value as tb_log_value
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Value


from Utils.Utils import time_str
from Replay.ExpReplay_Options import ExperienceReplay_Options
from Models.Models import get_torch_models as get_models

from Hierarchical.MazeOptions import MazeOptions
from Hierarchical.Primitive_Options import Primitive_Options
from Hierarchical.Random_Macro_Actions import Random_Macro_Actions

import Envs

parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--t-max", type=int, default=int(10e5))
parser.add_argument("--env", type=str, default="Maze-2-v1")
parser.add_argument("--logdir", type=str, default="Logs")
parser.add_argument("--name", type=str, default="nn")
parser.add_argument("--exp-replay-size", "--xp", type=int, default=int(5e4))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--model", type=str, default="")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--slow-render", action="store_true", default=False)
parser.add_argument("--epsilon-steps", "--eps-steps", type=int, default=int(7e5))
parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
parser.add_argument("--epsilon-finish", "--eps-finish", type=float, default=0.1)
parser.add_argument("--batch-size", "--batch", type=int, default=32)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target", type=int, default=100)
parser.add_argument("--count", action="store_true", default=False)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--exploration-steps", "--exp-steps", type=int, default=int(5e4))
parser.add_argument("--n-step", "--n", type=int, default=1)
parser.add_argument("--n-inc", action="store_true", default=False)
parser.add_argument("--n-max", type=int, default=10)
parser.add_argument("--plain-print", action="store_true", default=False)
parser.add_argument("--clip-value", type=float, default=5)
parser.add_argument("--no-tb", action="store_true", default=False)
parser.add_argument("--no-eval-images", action="store_true", default=False)
parser.add_argument("--eval-images-interval", type=int, default=4)
parser.add_argument("--tb-interval", type=int, default=10)
parser.add_argument("--debug-eval", action="store_true", default=False)
parser.add_argument("--cts-size", type=int, default=7)
parser.add_argument("--cts-conv", action="store_true", default=False)
parser.add_argument("--exp-bonus-save", type=float, default=0.75)
parser.add_argument("--clip-reward", action="store_true", default=False)
parser.add_argument("--options", type=str, default="Primitive")
parser.add_argument("--num-macros", type=int, default=10)
parser.add_argument("--max-macro-length", type=int, default=10)
parser.add_argument("--macro-seed", type=int, default=12)
parser.add_argument("--train-primitives", action="store_true", default=False)
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

while os.path.exists(LOGDIR):
    LOGDIR += "_"

print("Logging to:\n{}\n".format(LOGDIR))

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists("{}/logs".format(LOGDIR)):
    os.makedirs("{}/logs".format(LOGDIR))
if not os.path.exists("{}/evals".format(LOGDIR)):
    os.makedirs("{}/evals".format(LOGDIR))
if not os.path.exists("{}/exp_bonus".format(LOGDIR)):
    os.makedirs("{}/exp_bonus".format(LOGDIR))
if not os.path.exists("{}/visitations".format(LOGDIR)):
    os.makedirs("{}/visitations".format(LOGDIR))

with open("{}/settings.txt".format(LOGDIR), "w") as f:
    f.write(str(args))

# Seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu:
    torch.cuda.manual_seed_all(args.seed)

# N step start
args.n_start = args.n_step

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
args.primitive_actions = args.actions

# Experience Replay
replay = ExperienceReplay_Options(args.exp_replay_size)
if args.train_primitives:
    primitive_replay = ExperienceReplay_Options(args.exp_replay_size)

# Options
if args.options == "Random_Macros":
    macro_lengths = []
    ll = args.max_macro_length
    while ll > 1:
        macro_lengths.append(ll)
        ll = int(ll / 2)
    lengths = [m for m in macro_lengths]
    macro_lengths *= int(args.num_macros / len(macro_lengths))
    macro_lengths += macro_lengths[:args.num_macros - len(macro_lengths)]
    print("\n{} Macro actions of lengths: {}".format(len(macro_lengths), lengths))
    # print(macro_lengths, "\n")
    macro_lengths = sorted(macro_lengths)
    options = Random_Macro_Actions(num_primitive_actions=args.actions, lengths_of_macros=macro_lengths, seed=args.macro_seed, with_primitives=True)
    args.actions = len(macro_lengths)
    if not os.path.exists("{}/macros".format(LOGDIR)):
        os.makedirs("{}/macros".format(LOGDIR))
    with open("{}/macros/random_macros.txt".format(LOGDIR), "ab") as file:
        for macro in options.macros:
            np.savetxt(file, macro, delimiter=" ", fmt="%d", newline=" ")
            file.write(str.encode("\n"))
elif args.options == "Maze_Good":
    options = MazeOptions()
else:
    options = Primitive_Options()

print("\nOptions Being Used: {}\n".format(args.options))

# DQN
print("\nGetting Models.\n")
dqn = get_models(args.model)(actions=args.actions)
target_dqn = get_models(args.model)(actions=args.actions)

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
target_sync_T = 0

# Debug stuff
max_q_value = -1000
min_q_value = +1000
max_exp_bonus = 0

# Env specific stuff
Player_Positions = []
Player_Positions_With_Goals = []

# Async queue
log_queue = Queue()
# gif_queue = Queue()
eval_images = -args.t_max

if args.count:
    # Use half the env size
    # env_size = env.shape[0] * 7
    # cts_model_shape = (env_size // 2, env_size // 2)
    # Use a (14, 14) model anyway
    cts_model_shape = (args.cts_size, args.cts_size)
    print("\nCTS Model has size: " + str(cts_model_shape) + "\n")
    cts_model = CTS.DensityModel(frame_shape=cts_model_shape, context_functor=CTS.L_shaped_context, conv=args.cts_conv)
    os.makedirs("{}/cts_model".format(LOGDIR))


# class Variable(torch.autograd.Variable):

#     def __init__(self, data, *arguments, **kwargs):
#         if args.gpu:
#             data = data.cuda()
#             print(data)
#         super(Variable, self).__init__(data, *arguments, **kwargs)

# Multiprocessing logger
def logger(q, finished):
    configure("{}/tb".format(LOGDIR), flush_secs=30)
    # Crayon stuff
    # crayon_client = CrayonClient(hostname="localhost")
    # crayon_exp = crayon_client.create_experiment(NAME_DATE)
    while finished.value < 1:
        (name, value, step) = q.get(block=True)
        tb_log_value(name, value, step=step)
        # crayon_exp.add_scalar_value(name, value, step=step)


def log_value(name, value, step):
    log_queue.put((name, value, step))


def save_video(name, images):
    # name = name + ".mkv"
    name = name + ".gif"
    # TODO: Pad the images to macro block size
    imageio.mimsave(name, images, subrectangles=True)
    # imageio.mimsave(name, images, ffmpeg_log_level="error", quality=10)


def save_image(name, image):
    name = name + ".png"
    imageio.imsave(name, image)


# Methods
def environment_specific_stuff():
    if args.env.startswith("Maze"):
        player_pos = env.env.player_pos
        player_pos_with_goals = (player_pos[0], player_pos[1], env.env.maze[3, -1] != 2, env.env.maze[-1, -4] != 2, env.env.maze[-4, 0] != 2)
        Player_Positions.append(player_pos)
        Player_Positions_With_Goals.append(player_pos_with_goals)
        with open("{}/logs/Player_Positions_In_Maze.txt".format(LOGDIR), "a") as file:
            file.write(str(player_pos) + "\n")
        with open("{}/logs/Player_Positions_In_Maze_With_Goals.txt".format(LOGDIR), "a") as file:
            file.write(str(player_pos_with_goals) + "\n")

        # TODO: Move this out into a post-processing step
        if T % int(args.t_max / 2) == 0:
            # Make a gif of the positions
            for interval_size in [2, 10]:
                interval = int(args.t_max / interval_size)
                scaling = 2
                images = []
                for i in range(0, T, interval // 10):
                    canvas = np.zeros((env.env.maze.shape[0], env.env.maze.shape[1]))
                    for visit in Player_Positions[i: i + interval]:
                        canvas[visit] += 1
                    # Bit of a hack
                    if np.max(canvas) == 0:
                        break
                    gray_maze = canvas / (np.max(canvas) / scaling)
                    gray_maze = np.clip(gray_maze, 0, 1) * 255
                    images.append(gray_maze.astype(np.uint8))
                save_video("{}/visitations/Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), images)

                # We want to show visualisations for the agent depending on which goals they've visited as well
                # Keep it seperate from the other one
                colour_images = []
                for i in range(0, T, interval // 10):
                    canvas = np.zeros((env.env.maze.shape[0] * 3, env.env.maze.shape[1] * 3, 3))
                    maze_size = env.env.maze.shape[0]
                    for visit in Player_Positions_With_Goals[i: i + interval]:
                        px = visit[0]
                        py = visit[1]
                        g1 = visit[2]
                        g2 = visit[3]
                        g3 = visit[4]
                        if not g1 and not g2 and not g3:
                            # No Goals visited
                            canvas[px, py, :] += 1
                        elif g1 and not g2 and not g3:
                            # Only g1
                            canvas[px, py + maze_size, 0] += 1
                        elif not g1 and g2 and not g3:
                            # Only g2
                            canvas[px + maze_size, py + maze_size, 1] += 1
                        elif not g1 and not g2 and g3:
                            # Only g3
                            canvas[px + 2 * maze_size, py + maze_size, 2] += 1
                        elif g1 and g2 and not g3:
                            # g1 and g2
                            canvas[px, py + maze_size * 2, 0: 2] += 1
                        elif g1 and not g2 and g3:
                            # g1 and g3
                            canvas[px + maze_size, py + maze_size * 2, 0: 3: 2] += 1
                        elif not g1 and g2 and g3:
                            # g2 and g3
                            canvas[px + maze_size * 2, py + maze_size * 2, 1: 3] += 1
                        else:
                            # print("ERROR", g1, g2, g3)
                            pass
                    if np.max(canvas) == 0:
                        break
                    canvas = canvas / (np.max(canvas) / scaling)
                    # Colour the unvisited goals
                    # player_pos_with_goals = (player_pos[0], player_pos[1], env.maze[3, -1] != 2, env.maze[-1, -4] != 2, env.maze[-4, 0] != 2)
                    # only g1
                    canvas[maze_size - 1, maze_size + maze_size - 4, :] = 1
                    canvas[maze_size - 4, maze_size, :] = 1
                    # only g2
                    canvas[maze_size + 3, maze_size + maze_size - 1, :] = 1
                    canvas[maze_size + maze_size - 4, maze_size, :] = 1
                    # only g3
                    canvas[2 * maze_size + 3, maze_size + maze_size - 1, :] = 1
                    canvas[2 * maze_size + maze_size - 1, maze_size + maze_size - 4, :] = 1
                    # g1 and g2
                    canvas[maze_size - 4, 2 * maze_size] = 1
                    # g1 and g3
                    canvas[maze_size + maze_size - 1, 2 * maze_size + maze_size - 4] = 1
                    # g2 and g2
                    canvas[2 * maze_size + 3, 2 * maze_size + maze_size - 1] = 1
                    # Seperate the mazes
                    canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=0)
                    canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=1)
                    colour_maze = canvas

                    colour_maze = np.clip(colour_maze, 0, 1) * 255
                    colour_images.append(colour_maze.astype(np.uint8))
                save_video("{}/visitations/Goal_Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)


# TODO: Async this
def eval_agent(last=False):
    global epsilon
    # epsilon = 0

    global eval_images
    will_save_states = args.eval_images and (last or T - eval_images > (args.t_max // args.eval_images_interval))

    epsilons = [0, epsilon]

    for epsilon_value in epsilons:
        epsilon = epsilon_value
        terminated = False
        ep_reward = 0
        steps = 0
        state = env.reset()
        states = [state]
        Eval_Q_Values = []

        if will_save_states and args.debug_eval:
            env.env.debug_render(offline=True, debug_info={"Exp_Bonuses": 0, "CTS_Size": args.cts_size})
            debug_states = [pygame_image(env.env.surface)]

        while not terminated:
            action, q_vals = select_action(state, training=False)

            Eval_Q_Values.append(q_vals)

            if will_save_states:
                # Only do this stuff if we're actually gonna save it
                if args.debug_eval:
                    debug_dict = {}
                    debug_dict["Q_Values"] = q_vals
                    debug_dict["Max_Q_Value"] = max_q_value
                    debug_dict["Min_Q_Value"] = min_q_value
                    debug_dict["Action"] = action
                    if args.count:
                        exp_bonus, exp_info = exploration_bonus(state, training=False)
                        debug_dict["Max_Exp_Bonus"] = max_exp_bonus
                        debug_dict["Exp_Bonus"] = exp_bonus
                        cts_state = resize(state[:, :, 0], cts_model_shape, mode="F")
                        debug_dict["CTS_State"] = cts_state
                        debug_dict["CTS_PG"] = exp_info["Pixel_PG"]
                    # env.env.debug_render(debug_info=debug_dict, offline=True)
                    # debug_states.append(pygame_image(env.env.surface))
                states.append(state)

            option_terminated = False
            reward = 0
            options.choose_option(action)
            while not option_terminated:
                if will_save_states and args.debug_eval:
                    env.env.debug_render(debug_info=debug_dict, offline=True)
                    debug_states.append(pygame_image(env.env.surface))
                primitive_action, option_beta = options.act(env)
                state_new, option_reward, terminated, env_info = env.step(primitive_action)
                reward += option_reward
                steps += 1
                option_terminated = np.random.binomial(1, p=option_beta) == 1 or terminated

            ep_reward += reward
            # steps += 1

            state = state_new

        if will_save_states:
            if args.debug_eval:
                save_eval_states(debug_states, debug=True)
            else:
                save_eval_states(states)
            eval_images = T

        if epsilon != epsilons[-1]:
            with open("{}/logs/Eval_Q_Values__Epsilon_{}__T.txt".format(LOGDIR, epsilon), "ab") as file:
                np.savetxt(file, Eval_Q_Values[:], delimiter=" ", fmt="%f")
                file.write(str.encode("\n"))

            if args.tb:
                log_value("Eval/Epsilon_{:.2f}/Episode_Reward".format(epsilon), ep_reward, step=T)
                log_value("Eval/Epsilon_{:.2f}/Episode_Length".format(epsilon), steps, step=T)


def save_eval_states(states, debug=False):
    if debug:
        states = [s.swapaxes(0, 1) for s in states]
    else:
        states = [s[:, :, 0] * 255 for s in states]
    # gif_queue.put((states, "{}/evals/Greedy_Policy__T_{}__Ep_{}.gif".format(LOGDIR, T, episode)))
    # Don't really need the async for this since it is relatively infrequent
    save_video("{}/evals/Eval_Policy__Epsilon_{:.2f}__T_{}__Ep_{}".format(LOGDIR, epsilon, T, episode), states)


def exploration_bonus(state, training=True):
    bonus = 0
    extra_info = {}
    global max_exp_bonus
    if args.count:
        state = resize(state[:, :, 0], cts_model_shape, mode="F")
        # rho_old = np.exp(cts_model.update(state))
        rho_old, rho_old_pixels = cts_model.update(state)
        # rho_new = np.exp(cts_model.log_prob(state))
        rho_new, rho_new_pixels = cts_model.log_prob(state)
        pg = rho_new - rho_old
        pg_pixel = rho_new_pixels - rho_old_pixels
        extra_info["Pixel_PG"] = pg_pixel
        # pseudo_count = (rho_old * (1 - rho_new)) / (rho_new - rho_old)
        # Change to the pseudo_count they use in neural density models:
        pg = min(10, pg)
        pg = max(0, pg)
        pseudo_count = 1 / (np.expm1(pg))
        # pseudo_count = max(pseudo_count, 0)
        bonus = args.beta / sqrt(pseudo_count + 0.01)

        # Save suprising states after the first quarter of training
        if T > args.t_max / 4 and bonus >= args.exp_bonus_save * max_exp_bonus:
            debug_dict = {}
            debug_dict["Max_Exp_Bonus"] = max_exp_bonus
            debug_dict["Exp_Bonus"] = bonus
            debug_dict["CTS_State"] = state
            debug_dict["CTS_PG"] = pg_pixel
            env.env.debug_render(debug_info=debug_dict, offline=True)
            image = pygame_image(env.env.surface)
            image = image.swapaxes(0, 1)
            save_image("{}/exp_bonus/Ep_{}__T_{}__Bonus_{:.3f}".format(LOGDIR, episode, T, bonus), image)
        if training:
            Exploration_Bonus.append(bonus)
            if args.tb and T % args.tb_interval == 0:
                log_value("Count_Bonus", bonus, step=T)
    max_exp_bonus = max(max_exp_bonus, bonus)
    return bonus, extra_info


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


def end_of_episode_save():
    if args.count:
        with open("{}/cts_model/cts_model_end.pkl".format(LOGDIR), "wb") as file:
            pickle.dump(cts_model, file, pickle.HIGHEST_PROTOCOL)


def sync_target_network():
    for target, source in zip(target_dqn.parameters(), dqn.parameters()):
        target.data = source.data


def select_random_action():
    return np.random.choice(args.actions)


def select_action(state, training=True):
    dqn.eval()
    state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
    q_values = dqn(Variable(state, volatile=True)).cpu().data[0]
    q_values_numpy = q_values.numpy()

    global max_q_value
    global min_q_value
    # Decay it so that it reflects a recentish maximum q value
    max_q_value *= 0.9999
    min_q_value *= 0.9999
    max_q_value = max(max_q_value, np.max(q_values_numpy))
    min_q_value = min(min_q_value, np.min(q_values_numpy))

    if training:
        Q_Values.append(q_values_numpy)

        # Log the q values
        if args.tb:
            # crayon_exp.add_histogram_value("DQN/Q_Values", q_values_numpy.tolist(), tobuild=True, step=T)
            # q_val_dict = {}
            for index in range(args.actions):
                # q_val_dict["DQN/Action_{}_Q_Value".format(index)] = float(q_values_numpy[index])
                if T % args.tb_interval == 0:
                    log_value("DQN/Action_{}_Q_Value".format(index), q_values_numpy[index], step=T)
            # print(q_val_dict)
            # crayon_exp.add_scalar_dict(q_val_dict, step=T)

    if np.random.random() < epsilon:
        action = np.random.randint(low=0, high=args.actions)
    else:
        action = q_values.max(0)[1][0]  # Torch...

    return action, q_values_numpy


def explore():
    print("\nExploratory phase for {} steps.".format(args.exploration_steps))
    e_steps = 0
    while e_steps < args.exploration_steps:
        s = env.reset()
        s_t = s
        terminated = False
        while not terminated:
            print(e_steps, end="\r")
            a = select_random_action()
            option_terminated = False
            reward = 0
            steps = 0
            options.choose_option(a)
            while not option_terminated:
                primitive_action, option_beta = options.act(env)
                sn, option_reward, terminated, env_info = env.step(primitive_action)
                reward += option_reward
                steps += 1
                e_steps += 1
                option_terminated = np.random.binomial(1, p=option_beta) == 1 or terminated
                if args.train_primitives:
                    primitive_replay.Add_Exp(s_t, primitive_action, option_reward, sn, 1, terminated)
                s_t = sn

            if "Steps_Termination" in env_info:
                terminated = True
                break

            replay.Add_Exp(s, a, reward, sn, steps, terminated)
            s = sn

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
    dqn.eval()
    if args.n_inc:
        # Start at n and increase to n_max over T_MAX
        args.n_step = round(args.n_start + (T / args.t_max) * (args.n_max - args.n_start))
    # TODO: Use a named tuple for experience replay
    batch = replay.Sample_N(args.batch_size, args.n_step, args.gamma)
    if args.train_primitives:
        batch += primitive_replay.Sample_N(args.batch_size, args.n_step, args.gamma)
    columns = list(zip(*batch))

    states = Variable(torch.from_numpy(np.array(columns[0])).float().transpose_(1, 3))
    actions = Variable(torch.LongTensor(columns[1]))
    terminal_states = Variable(torch.FloatTensor(columns[5]))
    rewards = Variable(torch.FloatTensor(columns[2]))
    if args.clip_reward:
        rewards = torch.clamp(rewards, -1, 1)
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

    dqn.train()
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
    if args.tb and T % args.tb_interval == 0:
        log_value("DQN/Gradient_Norm", total_norm, step=T)
        log_value("DQN/Loss", l2_loss.data[0], step=T)
        log_value("DQN/TD_Error", td_error.mean().data[0], step=T)


######################
# Training procedure #
######################

# Start the async logger
finished_training = Value("i", 0)
p_log = Process(target=logger, args=(log_queue, finished_training), daemon=True)
# p_gif = Process(target=gif_saver, args=(gif_queue, finished_training), daemon=True)
p_log.start()
# p_gif.start()

explore()

sync_target_network()

start_time = time.time()

print("Training.\n\n\n")

while T < args.t_max:

    state = env.reset()
    state_timestep = state
    if args.render:
        debug_info = {}
        if args.count:
            debug_info["Exp_Bonuses"] = True
            debug_info["CTS_Size"] = args.cts_size
        env.env.debug_render(debug_info)
    if args.count:
        debug_info = {}
        debug_info["Exp_Bonuses"] = True
        debug_info["CTS_Size"] = args.cts_size
        env.env.debug_render(debug_info, offline=True)
    episode_finished = False
    episode_reward = 0
    episode_bonus_only_reward = 0
    episode_steps = 0

    epsilon = epsilon_schedule()

    start_of_episode()

    print_time()

    while not episode_finished:
        action, q_values = select_action(state)

        exp_bonus, exp_info = exploration_bonus(state)

        if args.render:
            debug_dict = {}
            debug_dict["Q_Values"] = q_values
            debug_dict["Max_Q_Value"] = max_q_value
            debug_dict["Min_Q_Value"] = min_q_value
            debug_dict["Action"] = action
            if args.count:
                debug_dict["Max_Exp_Bonus"] = max_exp_bonus
                debug_dict["Exp_Bonus"] = exp_bonus
                cts_state = resize(state[:, :, 0], cts_model_shape, mode="F")
                debug_dict["CTS_State"] = cts_state
                debug_dict["CTS_PG"] = exp_info["Pixel_PG"]

        option_terminated = False
        reward = 0
        steps = 0
        options.choose_option(action)
        while not option_terminated:
            if args.render:
                if args.slow_render:
                    time.sleep(0.1)
                env.env.debug_render(debug_dict)
            primitive_action, option_beta = options.act(env)
            state_new, option_reward, episode_finished, env_info = env.step(primitive_action)
            reward += option_reward
            episode_steps += 1
            steps += 1
            T += 1
            option_terminated = np.random.binomial(1, p=option_beta) == 1 or episode_finished
            environment_specific_stuff()
            if args.train_primitives:
                primitive_replay.Add_Exp(state_timestep, primitive_action, option_reward, state_new, 1, episode_finished)
            state_timestep = state_new

        # If the environment terminated because it reached a limit, we do not want the agent
        # to see that transition, since it makes the env non markovian wrt state
        if "Steps_Termination" in env_info:
            episode_finished = True
            break

        episode_reward += reward

        episode_bonus_only_reward += exp_bonus
        reward += exp_bonus

        Rewards.append(reward)
        States.append(state)
        States_Next.append(state_new)
        Actions.append(action)

        replay.Add_Exp(state, action, reward, state_new, steps, episode_finished)

        train_agent()

        if T - target_sync_T > args.target:
            sync_target_network()
            target_sync_T = T

        state = state_new

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

end_of_episode_save()

print("\nEvaluating Last Agent\n")
eval_agent(last=True)
print("Last Evaluation Finished")

finished_training.value = 10
time.sleep(5)

log_queue.close()
# gif_queue.close()
p_log.join(timeout=1)
# p_gif.join()
print("\nFinished\n")
