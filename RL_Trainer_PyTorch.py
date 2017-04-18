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

from Agent.DDQN_Agent import DDQN_Agent
from Exploration.Pseudo_Count import PseudoCount

class Trainer:

    def __init__(self, args, env):
        self.args = args
        self.env = env

        if self.args.gpu and not torch.cuda.is_available():
            print("CUDA unavailable! Switching to cpu only")

        # Seed everything
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

        print("\nGetting Models.\n")
        model = get_models(args.model)(actions=args.actions)
        self.agent = DDQN_Agent(model, args)

        if args.count:
            self.exp_model = PseudoCount(args)

        self.log_queue = Queue()

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



    # Multiprocessing logger
    def logger(self, q, finished):
        configure("{}/tb".format(self.args.log_path), flush_secs=30)
        while finished.value < 1:
            (name, value, step) = q.get(block=True)
            tb_log_value(name, value, step=step)


    def log_value(self, name, value, step):
        self.log_queue.put((name, value, step))


    def save_video(self, name, images):
        name = name + ".gif"
        # TODO: Pad the images to macro block size
        imageio.mimsave(name, images, subrectangles=True)


    def save_image(self, name, image):
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

                # We want to save the positions where the exploration bonus was high
                if not args.count:
                    continue
                colour_images = []
                for i in range(0, T, interval // 10):
                    canvas = np.zeros((env.env.maze.shape[0] * 3, env.env.maze.shape[1] * 3, 3))
                    maze_size = env.env.maze.shape[0]
                    for visit, bonus in zip(Player_Positions_With_Goals[i: i + interval], Exploration_Bonus[i: i + interval]):
                        relative_bonus = bonus / max_exp_bonus
                        px = visit[0]
                        py = visit[1]
                        g1 = visit[2]
                        g2 = visit[3]
                        g3 = visit[4]
                        # Assign the maximum bonus in that interval to the image
                        if not g1 and not g2 and not g3:
                            # No Goals visited
                            canvas[px, py, :] = max(relative_bonus, canvas[px, py, 0])
                        elif g1 and not g2 and not g3:
                            # Only g1
                            canvas[px, py + maze_size, 0] = max(relative_bonus, canvas[px, py + maze_size, 0])
                        elif not g1 and g2 and not g3:
                            # Only g2
                            canvas[px + maze_size, py + maze_size, 1] = max(relative_bonus, canvas[px + maze_size, py + maze_size, 1])
                        elif not g1 and not g2 and g3:
                            # Only g3
                            canvas[px + 2 * maze_size, py + maze_size, 2] = max(relative_bonus, canvas[px + 2 * maze_size, py + maze_size, 2])
                        elif g1 and g2 and not g3:
                            # g1 and g2
                            canvas[px, py + maze_size * 2, 0: 2] = max(relative_bonus, canvas[px, py + maze_size * 2, 0])
                        elif g1 and not g2 and g3:
                            # g1 and g3
                            canvas[px + maze_size, py + maze_size * 2, 0: 3: 2] = max(relative_bonus, canvas[px + maze_size, py + maze_size * 2, 0])
                        elif not g1 and g2 and g3:
                            # g2 and g3
                            canvas[px + maze_size * 2, py + maze_size * 2, 1: 3] = max(relative_bonus, canvas[px + maze_size * 2, py + maze_size * 2, 1])
                        else:
                            # print("ERROR", g1, g2, g3)
                            pass
                    canvas = np.clip(canvas, 0, 1)
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
                save_video("{}/exp_bonus/High_Exp_Bonus__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)


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
    save_video("{}/evals/Eval_Policy__Epsilon_{:.2f}__T_{}__Ep_{}".format(self.args.log_path, epsilon, T, episode), states)


    def exploration_bonus(self, state, training=True):

        bonus, extra_info = self.exp_model.bonus(state)

        # Save suprising states after the first quarter of training
        if self.T > self.args.t_max / 4 and bonus >= self.args.exp_bonus_save * self.max_exp_bonus:
            debug_dict = {}
            debug_dict["Max_Exp_Bonus"] = max_exp_bonus
            debug_dict["Exp_Bonus"] = bonus
            debug_dict["CTS_State"] = state
            debug_dict["CTS_PG"] = pg_pixel
            env.env.debug_render(debug_info=debug_dict, offline=True)
            image = pygame_image(env.env.surface)
            image = image.swapaxes(0, 1)
            save_image("{}/exp_bonus/Ep_{}__T_{}__Bonus_{:.3f}".format(self.args.log_path, self.episode, self.T, bonus), image)

        if training:
            self.Exploration_Bonus.append(bonus)
            if self.args.tb and self.T % self.args.tb_interval == 0:
                self.log_value("Count_Bonus", bonus, step=T)

        self.max_exp_bonus = max(self.max_exp_bonus, bonus)

        return bonus, extra_info


    def start_of_episode(self):
        if self.args.tb:
            self.log_value("Epsilon", self.epsilon, step=self.T)


    def end_of_episode(self):
        if self.args.tb:
            self.log_value("Episode_Reward", self.episode_reward, step=self.T)
            self.log_value("Episode_Bonus_Only_Reward", self.episode_bonus_only_reward, step=self.T)
            self.log_value("Episode_Length", self.episode_steps, step=self.T)


def save_values(self):

    if self.episode > self.Last_Ep_Logged:
        with open("{}/logs/Episode_Rewards.txt".format(self.args.log_path), "ab") as file:
            np.savetxt(file, self.Episode_Rewards[self.Last_Ep_Logged - 1:], delimiter=" ", fmt="%f")

        with open("{}/logs/Episode_Lengths.txt".format(self.args.log_path), "ab") as file:
            np.savetxt(file, self.Episode_Lengths[self.Last_Ep_Logged - 1:], delimiter=" ", fmt="%d")

        self.Last_Ep_Logged = self.episode

    if self.T > self.Last_T_Logged:
        with open("{}/logs/Q_Values_T.txt".format(self.args.log_path), "ab") as file:
            np.savetxt(file, self.Q_Values[self.Last_T_Logged - 1:], delimiter=" ", fmt="%f")
            file.write(str.encode("\n"))

        with open("{}/logs/DQN_Loss_T.txt".format(self.args.log_path), "ab") as file:
            np.savetxt(file, self.DQN_Loss[self.Last_T_Logged - 1:], delimiter=" ", fmt="%.10f")
            file.write(str.encode("\n"))

        if self.args.count:
            with open("{}/logs/Exploration_Bonus_T.txt".format(self.args.log_path), "ab") as file:
                np.savetxt(file, self.Exploration_Bonus[self.Last_T_Logged - 1:], delimiter=" ", fmt="%.10f")
                file.write(str.encode("\n"))

        self.Last_T_Logged = self.T


    def end_of_episode_save(self):
        if self.args.count:
            self.exp_model.save_model()


    def select_random_action(self):
        return np.random.choice(self.args.actions)


def select_action(state, training=True):
    action, extra_info = self.agent.act(state, self.epsilon, self.exp_model)

    if "Q_Values" in extra_info:

        q_values_numpy = extra_info["Q_Values"]

        # Decay it so that it reflects a recentish maximum q value
        self.max_q_value *= 0.9999
        self.min_q_value *= 0.9999
        self.max_q_value = max(self.max_q_value, np.max(q_values_numpy))
        self.min_q_value = min(self.min_q_value, np.min(q_values_numpy))

        if training:
            self.Q_Values.append(q_values_numpy)

            # Log the q values
            if self.gs.tb:
                for index in range(self.args.actions):
                    if self.T % self.args.tb_interval == 0:
                        self.log_value("DQN/Action_{}_Q_Value".format(index), q_values_numpy[index], step=self.T)

    return action, extra_info


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


    def print_time(self):
        if self.args.plain_print:
            print(self.T, end="\r")
        else:
            time_elapsed = time.time() - start_time
            time_left = time_elapsed * (self.args.t_max - self.T) / self.T
            # Just in case its over 100 days
            time_left = min(time_left, 60 * 60 * 24 * 100)
            last_reward = "N\A"
            if len(self.Episode_Rewards) > 10:
                last_reward = "{:.2f}".format(np.mean(self.Episode_Rewards[-10:-1]))
            print("\033[F\033[F\x1b[KEp: {:,}, T: {:,}/{:,}, Epsilon: {:.2f}, Reward: {}, \n\x1b[KElapsed: {}, Left: {}\n".format(self.episode, self.T, self.args.t_max, self.epsilon, self.last_reward, time_str(time_elapsed), time_str(time_left)), " " * 10, end="\r")


    def epsilon_schedule(self):
        return args.epsilon_finish + (args.epsilon_start - args.epsilon_finish) * max(((args.epsilon_steps - T) / args.epsilon_steps), 0)


def train_agent():
    dqn.eval()
    if args.n_inc:
        # Start at n and increase to n_max over T_MAX
        args.n_step = round(args.n_start + (T / args.t_max) * (args.n_max - args.n_start))
    # TODO: Use a named tuple for experience replay
    batch = replay.Sample_N(args.batch_size, args.n_step, args.gamma)
    if args.train_primitives:
        batch = batch + primitive_replay.Sample_N(args.batch_size, args.n_step, args.gamma)
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

    q_value_targets = (Variable(torch.ones(terminal_states.size()[0])) - terminal_states)
    inter = Variable(torch.ones(terminal_states.size()[0]) * args.gamma)
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
                if np.all(state_timestep == state):
                    # We add in the exploration bonus out of state
                    option_reward += exp_bonus
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
