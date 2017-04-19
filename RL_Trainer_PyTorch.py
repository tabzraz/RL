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
from Monitoring.Env_Wrapper import EnvWrapper

class Trainer:

    def __init__(self, args, env):
        self.args = args
        self.env = EnvWrapper(env)

        if args.gpu and not torch.cuda.is_available():
            print("CUDA unavailable! Switching to cpu only")

        # Seed everything
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

        print("\nGetting Models.\n")
        model = get_models(args.model)
        self.agent = DDQN_Agent(model, args)

        if args.count:
            self.exp_model = PseudoCount(args)

        self.log_queue = Queue()

        self.eval_video_T = -self.args.t_max

        # Stuff to log
        self.Q_Values = []
        self.Episode_Rewards = []
        self.Episode_Bonus_Only_Rewards = []
        self.Episode_Lengths = []
        self.DQN_Loss = []
        self.DQN_Grad_Norm = []
        self.Exploration_Bonus = []

        self.Last_T_Logged = 1
        self.Last_Ep_Logged = 1

        # Variables and stuff
        self.T = 1
        self.episode = 1
        self.epsilon = 1
        self.episode_reward = 0
        self.episode_bonus_only_reward = 0
        self.epsiode_steps = 0

        # Debug stuff
        self.max_q_value = -1000
        self.min_q_value = +1000
        self.max_exp_bonus = 0

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

# # Methods
# def environment_specific_stuff():
#     if args.env.startswith("Maze"):
#         player_pos = env.env.player_pos
#         player_pos_with_goals = (player_pos[0], player_pos[1], env.env.maze[3, -1] != 2, env.env.maze[-1, -4] != 2, env.env.maze[-4, 0] != 2)
#         Player_Positions.append(player_pos)
#         Player_Positions_With_Goals.append(player_pos_with_goals)
#         with open("{}/logs/Player_Positions_In_Maze.txt".format(LOGDIR), "a") as file:
#             file.write(str(player_pos) + "\n")
#         with open("{}/logs/Player_Positions_In_Maze_With_Goals.txt".format(LOGDIR), "a") as file:
#             file.write(str(player_pos_with_goals) + "\n")

#         # TODO: Move this out into a post-processing step
#         if T % int(args.t_max / 2) == 0:
#             # Make a gif of the positions
#             for interval_size in [2, 10]:
#                 interval = int(args.t_max / interval_size)
#                 scaling = 2
#                 images = []
#                 for i in range(0, T, interval // 10):
#                     canvas = np.zeros((env.env.maze.shape[0], env.env.maze.shape[1]))
#                     for visit in Player_Positions[i: i + interval]:
#                         canvas[visit] += 1
#                     # Bit of a hack
#                     if np.max(canvas) == 0:
#                         break
#                     gray_maze = canvas / (np.max(canvas) / scaling)
#                     gray_maze = np.clip(gray_maze, 0, 1) * 255
#                     images.append(gray_maze.astype(np.uint8))
#                 save_video("{}/visitations/Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), images)

#                 # We want to show visualisations for the agent depending on which goals they've visited as well
#                 # Keep it seperate from the other one
#                 colour_images = []
#                 for i in range(0, T, interval // 10):
#                     canvas = np.zeros((env.env.maze.shape[0] * 3, env.env.maze.shape[1] * 3, 3))
#                     maze_size = env.env.maze.shape[0]
#                     for visit in Player_Positions_With_Goals[i: i + interval]:
#                         px = visit[0]
#                         py = visit[1]
#                         g1 = visit[2]
#                         g2 = visit[3]
#                         g3 = visit[4]
#                         if not g1 and not g2 and not g3:
#                             # No Goals visited
#                             canvas[px, py, :] += 1
#                         elif g1 and not g2 and not g3:
#                             # Only g1
#                             canvas[px, py + maze_size, 0] += 1
#                         elif not g1 and g2 and not g3:
#                             # Only g2
#                             canvas[px + maze_size, py + maze_size, 1] += 1
#                         elif not g1 and not g2 and g3:
#                             # Only g3
#                             canvas[px + 2 * maze_size, py + maze_size, 2] += 1
#                         elif g1 and g2 and not g3:
#                             # g1 and g2
#                             canvas[px, py + maze_size * 2, 0: 2] += 1
#                         elif g1 and not g2 and g3:
#                             # g1 and g3
#                             canvas[px + maze_size, py + maze_size * 2, 0: 3: 2] += 1
#                         elif not g1 and g2 and g3:
#                             # g2 and g3
#                             canvas[px + maze_size * 2, py + maze_size * 2, 1: 3] += 1
#                         else:
#                             # print("ERROR", g1, g2, g3)
#                             pass
#                     if np.max(canvas) == 0:
#                         break
#                     canvas = canvas / (np.max(canvas) / scaling)
#                     # Colour the unvisited goals
#                     # player_pos_with_goals = (player_pos[0], player_pos[1], env.maze[3, -1] != 2, env.maze[-1, -4] != 2, env.maze[-4, 0] != 2)
#                     # only g1
#                     canvas[maze_size - 1, maze_size + maze_size - 4, :] = 1
#                     canvas[maze_size - 4, maze_size, :] = 1
#                     # only g2
#                     canvas[maze_size + 3, maze_size + maze_size - 1, :] = 1
#                     canvas[maze_size + maze_size - 4, maze_size, :] = 1
#                     # only g3
#                     canvas[2 * maze_size + 3, maze_size + maze_size - 1, :] = 1
#                     canvas[2 * maze_size + maze_size - 1, maze_size + maze_size - 4, :] = 1
#                     # g1 and g2
#                     canvas[maze_size - 4, 2 * maze_size] = 1
#                     # g1 and g3
#                     canvas[maze_size + maze_size - 1, 2 * maze_size + maze_size - 4] = 1
#                     # g2 and g2
#                     canvas[2 * maze_size + 3, 2 * maze_size + maze_size - 1] = 1
#                     # Seperate the mazes
#                     canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=0)
#                     canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=1)
#                     colour_maze = canvas

#                     colour_maze = np.clip(colour_maze, 0, 1) * 255
#                     colour_images.append(colour_maze.astype(np.uint8))
#                 save_video("{}/visitations/Goal_Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)

#                 # We want to save the positions where the exploration bonus was high
#                 if not args.count:
#                     continue
#                 colour_images = []
#                 for i in range(0, T, interval // 10):
#                     canvas = np.zeros((env.env.maze.shape[0] * 3, env.env.maze.shape[1] * 3, 3))
#                     maze_size = env.env.maze.shape[0]
#                     for visit, bonus in zip(Player_Positions_With_Goals[i: i + interval], Exploration_Bonus[i: i + interval]):
#                         relative_bonus = bonus / max_exp_bonus
#                         px = visit[0]
#                         py = visit[1]
#                         g1 = visit[2]
#                         g2 = visit[3]
#                         g3 = visit[4]
#                         # Assign the maximum bonus in that interval to the image
#                         if not g1 and not g2 and not g3:
#                             # No Goals visited
#                             canvas[px, py, :] = max(relative_bonus, canvas[px, py, 0])
#                         elif g1 and not g2 and not g3:
#                             # Only g1
#                             canvas[px, py + maze_size, 0] = max(relative_bonus, canvas[px, py + maze_size, 0])
#                         elif not g1 and g2 and not g3:
#                             # Only g2
#                             canvas[px + maze_size, py + maze_size, 1] = max(relative_bonus, canvas[px + maze_size, py + maze_size, 1])
#                         elif not g1 and not g2 and g3:
#                             # Only g3
#                             canvas[px + 2 * maze_size, py + maze_size, 2] = max(relative_bonus, canvas[px + 2 * maze_size, py + maze_size, 2])
#                         elif g1 and g2 and not g3:
#                             # g1 and g2
#                             canvas[px, py + maze_size * 2, 0: 2] = max(relative_bonus, canvas[px, py + maze_size * 2, 0])
#                         elif g1 and not g2 and g3:
#                             # g1 and g3
#                             canvas[px + maze_size, py + maze_size * 2, 0: 3: 2] = max(relative_bonus, canvas[px + maze_size, py + maze_size * 2, 0])
#                         elif not g1 and g2 and g3:
#                             # g2 and g3
#                             canvas[px + maze_size * 2, py + maze_size * 2, 1: 3] = max(relative_bonus, canvas[px + maze_size * 2, py + maze_size * 2, 1])
#                         else:
#                             # print("ERROR", g1, g2, g3)
#                             pass
#                     canvas = np.clip(canvas, 0, 1)
#                     # Colour the unvisited goals
#                     # player_pos_with_goals = (player_pos[0], player_pos[1], env.maze[3, -1] != 2, env.maze[-1, -4] != 2, env.maze[-4, 0] != 2)
#                     # only g1
#                     canvas[maze_size - 1, maze_size + maze_size - 4, :] = 1
#                     canvas[maze_size - 4, maze_size, :] = 1
#                     # only g2
#                     canvas[maze_size + 3, maze_size + maze_size - 1, :] = 1
#                     canvas[maze_size + maze_size - 4, maze_size, :] = 1
#                     # only g3
#                     canvas[2 * maze_size + 3, maze_size + maze_size - 1, :] = 1
#                     canvas[2 * maze_size + maze_size - 1, maze_size + maze_size - 4, :] = 1
#                     # g1 and g2
#                     canvas[maze_size - 4, 2 * maze_size] = 1
#                     # g1 and g3
#                     canvas[maze_size + maze_size - 1, 2 * maze_size + maze_size - 4] = 1
#                     # g2 and g2
#                     canvas[2 * maze_size + 3, 2 * maze_size + maze_size - 1] = 1
#                     # Seperate the mazes
#                     canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=0)
#                     canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=1)
#                     colour_maze = canvas

#                     colour_maze = np.clip(colour_maze, 0, 1) * 255
#                     colour_images.append(colour_maze.astype(np.uint8))
#                 save_video("{}/exp_bonus/High_Exp_Bonus__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)

    def eval_agent(self, last=False):
        env = self.env

        will_save_states = self.args.eval_images and (last or self.T - self.eval_video_T > (self.args.t_max // self.args.eval_images_interval))

        epsilons = [0, self.epsilon]

        for epsilon_value in epsilons:
            self.epsilon = epsilon_value
            terminated = False
            ep_reward = 0
            steps = 0
            state = env.reset()
            states = []
            Eval_Q_Values = []

            while not terminated:
                action, action_info = self.select_action(state, training=False)

                if "Q_Values" in action_info:
                    Eval_Q_Values.append(action_info["Q_Values"])

                if will_save_states:
                    # Only do this stuff if we're actually gonna save it
                    debug_state = state
                    if self.args.debug_eval:
                        debug_info = {}
                        debug_info.update(action_info)
                        if self.args.count:
                            exp_bonus, exp_info = self.exploration_bonus(state, training=False)
                            debug_info.update(exp_info)
                        debug_state = env.debug_render(debug_info, mode="rgb_array")

                    states.append(debug_state)

                state_new, reward, terminated, env_info = env.step(action)
                steps += 1

                ep_reward += reward

                state = state_new

            if will_save_states:
                save_video("{}/evals/Eval_Policy__Epsilon_{:.2f}__T_{}__Ep_{}".format(self.args.log_path, epsilon_value, self.T, self.episode), states)
                self.eval_video_T = self.T

            if epsilon_value != epsilons[-1]:
                with open("{}/logs/Eval_Q_Values__Epsilon_{}__T.txt".format(self.args.log_path, epsilon_value), "ab") as file:
                    np.savetxt(file, Eval_Q_Values[:], delimiter=" ", fmt="%f")
                    file.write(str.encode("\n"))

                if self.args.tb:
                    self.log_value("Eval/Epsilon_{:.2f}/Episode_Reward".format(epsilon_value), ep_reward, step=self.T)
                    self.log_value("Eval/Epsilon_{:.2f}/Episode_Length".format(epsilon_value), steps, step=self.T)

    def exploration_bonus(self, state, training=True):

        bonus, extra_info = self.exp_model.bonus(state)

        # Save suprising states after the first quarter of training
        if self.T > self.args.t_max / 4 and bonus >= self.args.exp_bonus_save * self.max_exp_bonus:
            debug_dict = {}
            debug_dict["Max_Exp_Bonus"] = max_exp_bonus
            debug_dict["Exp_Bonus"] = bonus
            debug_dict["CTS_State"] = state
            debug_dict["CTS_PG"] = pg_pixel
            image = debug_render(state, debug_dict)
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

    def select_action(self, state, training=True):
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
                if self.args.tb:
                    for index in range(self.args.actions):
                        if self.T % self.args.tb_interval == 0:
                            self.log_value("DQN/Action_{}_Q_Value".format(index), q_values_numpy[index], step=self.T)

        return action, extra_info

    def explore(self):
        env = self.env
        print("\nExploratory phase for {} steps.".format(args.exploration_steps))
        e_steps = 0
        while e_steps < self.args.exploration_steps:
            s = env.reset()
            s_t = s
            terminated = False
            while not terminated:
                print(e_steps, end="\r")
                a = self.select_random_action()

                sn, reward, terminated, env_info = env.step(a)
                e_steps += 1

                if "Steps_Termination" in env_info:
                    terminated = True
                    break

                self.agent.experience(s, a, reward, sn, 1, terminated)
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

    def train_agent(self):

        train_info = self.agent.train()

        if self.args.tb and self.T % self.args.tb_interval == 0:
            if "Norm" in train_info:
                self.log_value("DQN/Gradient_Norm", train_info["Norm"], step=self.T)
            if "Loss" in train_info:
                self.log_value("DQN/Loss", train_info["Loss"], step=self.T)
            if "TD_Error" in train_info:
                self.log_value("DQN/TD_Error", train_info["TD_Error"], step=self.T)


######################
# Training procedure #
######################

    def train(self):

        # Convenience
        env = self.env

        # Start the async logger
        finished_training = Value("i", 0)
        p_log = Process(target=self.logger, args=(self.log_queue, finished_training), daemon=True)
        p_log.start()

        self.explore()

        start_time = time.time()

        print("Training.\n\n\n")

        while self.T < self.args.t_max:

            state = env.reset()

            episode_finished = False
            self.episode_reward = 0
            self.episode_bonus_only_reward = 0
            self.episode_steps = 0

            self.epsilon = self.epsilon_schedule()

            self.start_of_episode()

            self.print_time()

            while not episode_finished:
                action, action_info = self.select_action(state)

                exp_bonus, exp_info = self.exploration_bonus(state)

                if self.args.render:
                    debug_info = {}
                    debug_info.update(action_info)
                    debug_info.update(exp_info)
                    if self.args.slow_render:
                        time.sleep(0.1)
                    self.env.debug_render(debug_info)

                state_new, reward, episode_finished, env_info = self.env.step(action)
                self.T += 1

                # If the environment terminated because it reached a limit, we do not want the agent
                # to see that transition, since it makes the env non markovian wrt state
                if "Steps_Termination" in env_info:
                    episode_finished = True
                    break

                self.episode_reward += reward

                self.episode_bonus_only_reward += exp_bonus
                # TODO: Update this to store pseudo-counts seperately
                reward += exp_bonus

                self.agent.experience(state, action, reward, state_new, steps, episode_finished)

                self.train_agent()

                state = state_new

                if not self.args.plain_print:
                    print("\x1b[K" + "." * ((self.episode_steps // 20) % 40), end="\r")

            self.eval_agent()

            self.episode += 1
            self.Episode_Rewards.append(self.episode_reward)
            self.Episode_Lengths.append(self.episode_steps)
            self.Episode_Bonus_Only_Rewards.append(self.episode_bonus_only_reward)

            self.end_of_episode()

            self.save_values()

        self.end_of_episode_save()

        print("\nEvaluating Last Agent\n")
        self.eval_agent(last=True)
        print("Last Evaluation Finished")

        # Close out the logging queue
        print("\nClosing queue")
        finished_training.value = 10
        time.sleep(5)
        self.log_queue.close()
        p_log.join(timeout=1)

        print("\nFinished\n")
