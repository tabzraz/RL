import time
import numpy as np

import torch

import imageio

from tensorboard_logger import configure
from tensorboard_logger import log_value as tb_log_value
from multiprocessing import Queue as Queue_MP
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
import queue

from pympler import asizeof

from Utils.Utils import time_str

from Agent.DDQN_Agent import DDQN_Agent
from Agent.Goal_DQN_Agent import Goal_DQN_Agent
from Agent.TabQ_Agent import TabQ_Agent
from Exploration.Pseudo_Count import PseudoCount
from Monitoring.Env_Wrapper import EnvWrapper


class Trainer:

    def __init__(self, args, env, eval_env):
        self.args = args
        self.env = EnvWrapper(env, True, args)
        self.eval_env = EnvWrapper(eval_env, True, args)

        if args.gpu and not torch.cuda.is_available():
            print("CUDA unavailable! Switching to cpu only")

        # Seed everything
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

        self.exp_model = None
        if args.count:
            self.exp_model = PseudoCount(args)

        if args.tabular:
            self.agent = TabQ_Agent(args, self.exp_model)
        elif args.goal_dqn:
            self.agent = Goal_DQN_Agent(args, self.exp_model)
        else:
            self.agent = DDQN_Agent(args, self.exp_model)

        self.log_queue = Queue_MP()

        self.eval_video_T = -self.args.t_max
        self.training_video_T = -self.args.t_max

        self.eval_T = 0

        # Frontier stuff
        self.frontier_T = 0
        self.frontier_images = []

        # Visualisations
        self.vis_T = 0
        self.bonus_images = []
        self.trained_on_states_images = []
        self.replay_states_images = []
        self.player_visits_images = []

        # Stuff to log
        self.Q_Values = []
        self.Episode_Rewards = []
        self.Episode_Bonus_Only_Rewards = []
        self.Episode_Lengths = []
        self.DQN_Loss = []
        self.DQN_Grad_Norm = []
        self.Exploration_Bonus = []
        self.Player_Positions = []
        self.Visited_States = set()
        self.Trained_On_States = []
        self.Eval_Rewards = []
        self.Action_State_Counts = []

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
        self.max_exp_bonus = 0.000001

    # Multiprocessing logger
    def logger(self, q, finished):
        configure("{}/tb".format(self.args.log_path), flush_secs=30)
        while finished.value < 1:
            try:
                (name, value, step) = q.get(block=False)
                tb_log_value(name, value, step=step)
            except queue.Empty:
                pass
        print("Logging loop closed")

    def log_value(self, name, value, step):
        self.log_queue.put((name, value, step))

    def save_video(self, name, images):
        name = name + ".gif"
        # TODO: Pad the images to macro block size
        images = [image.astype(np.uint8) for image in images]
        # Subrectangles to make the gifs smaller
        imageio.mimsave(name, images, subrectangles=True)
        # imageio.mimsave(name, images)

    def save_image(self, name, image):
        name = name + ".png"
        image = image.astype(np.uint8)
        imageio.imsave(name, image)

    def eval_agent(self, last=False):
        self.eval_T = self.T
        env = self.eval_env

        will_save_states = self.args.eval_images and (last or self.T - self.eval_video_T > (self.args.t_max // self.args.eval_images_interval))

        epsilons = [0.05]

        for epsilon_value in epsilons:
            # self.epsilon = epsilon_value
            terminated = False
            ep_reward = 0
            steps = 0
            state = env.reset()
            states = []
            Eval_Q_Values = []

            while not terminated:
                action, action_info = self.select_action(state, epsilon_value, training=False)

                if "Q_Values" in action_info:
                    Eval_Q_Values.append(action_info["Q_Values"])

                if will_save_states:
                    debug_info = {}
                    debug_info.update(action_info)
                    if self.args.count:
                        exp_bonus, exp_info = self.exploration_bonus(state, action, training=False)
                        exp_info["Max_Bonus"] = self.max_exp_bonus
                        debug_info.update(exp_info)
                    debug_state = env.debug_render(debug_info, mode="rgb_array")

                    states.append(debug_state)

                state_new, reward, terminated, env_info = env.step(action)
                steps += 1

                ep_reward += reward

                state = state_new

            if will_save_states:
                self.save_video("{}/evals/Eval_Policy__Epsilon_{:.2f}__T_{}__Ep_{}".format(self.args.log_path, epsilon_value, self.T, self.episode), states)
                self.eval_video_T = self.T

            self.Eval_Rewards.append(ep_reward)
            with open("{}/logs/Eval_Q_Values__Epsilon_{}__T.txt".format(self.args.log_path, epsilon_value), "ab") as file:
                np.savetxt(file, Eval_Q_Values[:], delimiter=" ", fmt="%f")
                file.write(str.encode("\n"))

            with open("{}/logs/Eval_Rewards__Epsilon_{}.txt".format(self.args.log_path, epsilon_value), "a") as file:
                file.write("{}\n".format(ep_reward))

            if self.args.tb:
                self.log_value("Eval/Epsilon_{:.2f}/Episode_Reward".format(epsilon_value), ep_reward, step=self.T)
                self.log_value("Eval/Epsilon_{:.2f}/Episode_Length".format(epsilon_value), steps, step=self.T)

    def exploration_bonus(self, state, action, training=True):

        bonus = 0
        extra_info = {}

        if self.args.count:
            if not self.args.count_state_action:
                action = 0
            state = state[:, :, -1:]
            bonus, extra_info = self.exp_model.bonus(state, action, dont_remember=not training)

            # TODO: Log for different actions
            if training:
                self.Exploration_Bonus.append(bonus)
                if self.args.tb and self.T % self.args.tb_interval == 0:
                    self.log_value("Count/Bonus", bonus, step=self.T)
                    self.log_value("Count/PseudoCounts", extra_info["Pseudo_Count"], step=self.T)
                    self.log_value("Count/Density", extra_info["Density"], step=self.T)

            self.max_exp_bonus = max(self.max_exp_bonus, bonus)
            extra_info["Max_Bonus"] = self.max_exp_bonus

            # Save suprising states after the first quarter of training
            if self.T > self.args.t_max / 4 and bonus >= self.args.exp_bonus_save * self.max_exp_bonus:
                image = self.env.debug_render(extra_info, mode="rgb_array")
                self.save_image("{}/exp_bonus/Ep_{}__T_{}__Action_{}__Bonus_{:.3f}".format(self.args.log_path, self.episode, self.T, action, bonus), image)

        return bonus, extra_info

    def start_of_episode(self):
        if self.args.tb:
            self.log_value("Epsilon", self.epsilon, step=self.T)

    def end_of_episode(self):
        self.agent.end_of_trajectory()
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

            with open("{}/logs/Action_Counts.txt".format(self.args.log_path), "ab") as file:
                np.savetxt(file, self.Action_State_Counts[self.Last_T_Logged - 1:], delimiter=" ", fmt="%d")
                file.write(str.encode("\n"))

            with open("{}/logs/DQN_Loss_T.txt".format(self.args.log_path), "ab") as file:
                np.savetxt(file, self.DQN_Loss[self.Last_T_Logged - 1:], delimiter=" ", fmt="%.10f")
                file.write(str.encode("\n"))

            if self.args.count:
                with open("{}/logs/Exploration_Bonus_T.txt".format(self.args.log_path), "ab") as file:
                    np.savetxt(file, self.Exploration_Bonus[self.Last_T_Logged - 1:], delimiter=" ", fmt="%.10f")
                    file.write(str.encode("\n"))

            self.Last_T_Logged = self.T

    def end_of_training_save(self):
        if self.args.count:
            self.exp_model.save_model()
            if len(self.bonus_images) > 0:
                self.save_video("{}/exp_bonus/Bonuses__Interval_{}".format(self.args.log_path, self.args.interval_size, self.T), self.bonus_images)
            if len(self.frontier_images) > 0:
                self.save_video("{}/exp_bonus/Frontier__Interval_{}".format(self.args.log_path, self.args.frontier_interval), self.frontier_images)
        if len(self.player_visits_images) > 0:
            self.save_video("{}/visitations/Goal_Visits__Interval_{}".format(self.args.log_path, self.args.interval_size), self.player_visits_images)
        if len(self.replay_states_images) > 0:
            self.save_video("{}/visitations/Xp_Replay_{:,}__Interval_{}".format(self.args.log_path, self.args.exp_replay_size, self.args.interval_size), self.replay_states_images)
        if self.args.log_trained_on_states:
            if len(self.trained_on_states_images) > 0:
                self.save_video("{}/visitations/Trained_States_In_Xp_{:,}__Interval_{}".format(self.args.log_path, self.args.exp_replay_size, self.args.interval_size), self.trained_on_states_images)

    def select_random_action(self):
        return np.random.choice(self.args.actions)

    def select_action(self, state, epsilon, training=True):
        action, extra_info = self.agent.act(state, epsilon, self.exp_model)
        extra_info["Epsilon"] = epsilon

        if "Q_Values" in extra_info:

            q_values_numpy = extra_info["Q_Values"]

            # Decay it so that it reflects a recentish maximum q value
            self.max_q_value *= 0.9999
            self.min_q_value *= 0.9999
            self.max_q_value = max(self.max_q_value, np.max(q_values_numpy))
            self.min_q_value = min(self.min_q_value, np.min(q_values_numpy))
            extra_info["Max_Q_Value"] = self.max_q_value
            extra_info["Min_Q_Value"] = self.min_q_value

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
        print("\nExploratory phase for {} steps.".format(self.args.exploration_steps))
        e_steps = 0
        while e_steps < self.args.exploration_steps:
            s = env.reset()
            terminated = False
            while not terminated:
                print(e_steps, end="\r")
                a = self.select_random_action()

                # Prime the exploration model a little
                # bonus = 0
                # if self.args.count:
                #     bonus, _ = self.exp_model.bonus(s)
                #     self.max_exp_bonus = max(self.max_exp_bonus, bonus)

                sn, reward, terminated, env_info = env.step(a)
                e_steps += 1

                if "Steps_Termination" in env_info:
                    terminated = True
                    break

                self.agent.experience(s, a, reward, sn, 1, terminated, pseudo_reward=0, exploring=True)
                s = sn
            self.agent.end_of_trajectory()

        print("Exploratory phase finished. Starting learning.\n")

    def print_time(self):
        if self.args.plain_print:
            print(self.T, end="\r")
        else:
            time_elapsed = time.time() - self.start_time
            time_left = time_elapsed * (self.args.t_max - self.T) / self.T
            # Just in case its over 100 days
            time_left = min(time_left, 60 * 60 * 24 * 100)
            last_reward = "N\A"
            if len(self.Episode_Rewards) > 5:
                last_reward = "{:.2f}".format(np.mean(self.Episode_Rewards[-5:-1]))
            print("\033[F\033[F\x1b[KEp: {:,}, T: {:,}/{:,}, Epsilon: {:.2f}, Reward: {}, \n\x1b[KElapsed: {}, Left: {}\n".format(self.episode, self.T, self.args.t_max, self.epsilon, last_reward, time_str(time_elapsed), time_str(time_left)), " " * 10, end="\r")

    def epsilon_schedule(self):
        args = self.args
        return args.epsilon_finish + (args.epsilon_start - args.epsilon_finish) * max(((args.epsilon_steps - self.T) / args.epsilon_steps), 0)

    def train_agent(self):

        train_info = self.agent.train()

        if self.args.tb and self.T % self.args.tb_interval == 0:
            if "Norm" in train_info:
                self.log_value("DQN/Gradient_Norm", train_info["Norm"], step=self.T)
            if "Loss" in train_info:
                self.log_value("DQN/Loss", train_info["Loss"], step=self.T)
                self.DQN_Loss.append(train_info["Loss"])
            if "TD_Error" in train_info:
                self.log_value("DQN/TD_Error", train_info["TD_Error"], step=self.T)

        if self.args.log_trained_on_states:
            if "States" in train_info:
                self.Trained_On_States += [self.env.state_to_player_pos(s) for s in train_info["States"]]

    def visitations(self):
        player_pos = self.env.log_visitation()
        self.Player_Positions.append(player_pos)
        self.Visited_States.add(player_pos)
        if self.args.tb and self.T % self.args.tb_interval == 0:
            self.log_value("States_Visited", len(self.Visited_States), step=self.T)

    def frontier_vis(self):
        if self.args.count and self.T - self.frontier_T > self.args.t_max // self.args.frontier_interval:
            image = self.env.frontier(self.exp_model, max_bonus=self.max_exp_bonus)
            if image is not None:
                self.frontier_images.append(image)
            self.frontier_T = self.T

    def bonus_landscape(self):
        if self.args.count:
            entries = self.args.t_max // self.args.interval_size
            image = self.env.explorations(self.Player_Positions[-entries:], self.Exploration_Bonus[-entries:], self.max_exp_bonus)
            if image is not None:
                self.bonus_images.append(image)

    def trained_on_states(self):
        if self.args.log_trained_on_states:
            entries = self.args.t_max // self.args.interval_size
            image = self.env.trained_on_states(self.Trained_On_States[-entries * self.args.batch_size:])
            if image is not None:
                self.trained_on_states_images.append(image)

    def replay_states(self):
        image = self.env.xp_replay_states(self.Player_Positions[-self.args.exp_replay_size:])
        if image is not None:
            self.replay_states_images.append(image)

    def player_visits(self):
        entries = self.args.t_max // self.args.interval_size
        image = self.env.visitations(self.Player_Positions[-entries:])
        if image is not None:
            self.player_visits_images.append(image)

    def visualisations(self):
        if self.args.frontier:
            self.frontier_vis()
        if self.T - self.vis_T >= self.args.t_max // self.args.interval_size:
            # print("\n\n\n", self.T, len(self.Trained_On_States), len(self.Player_Positions),"\n\n\n")
            self.bonus_landscape()
            self.trained_on_states()
            self.replay_states()
            self.player_visits()

            self.vis_T = self.T

            # Save some ram
            entries = self.args.t_max // self.args.interval_size
            self.Trained_On_States = self.Trained_On_States[-entries * self.args.batch_size:]
            self.Player_Positions = self.Player_Positions[-self.args.exp_replay_size:]


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

        self.start_time = time.time()

        print("Training.\n\n\n")

        while self.T < self.args.t_max:

            state = env.reset()

            episode_finished = False
            self.episode_reward = 0
            self.episode_bonus_only_reward = 0
            self.episode_steps = 0

            self.epsilon = self.epsilon_schedule()
            new_epsilon = self.epsilon

            self.start_of_episode()

            self.print_time()

            will_save_states = self.args.eval_images and self.T - self.training_video_T > (self.args.t_max // self.args.eval_images_interval)
            video_states = []

            while not episode_finished:
                # TODO: Cleanup
                # new_epsilon = self.epsilon
                if self.args.count_epsilon:
                    exp_bonus, exp_info = self.exploration_bonus(state, action=0)
                    if self.args.epsilon_decay:
                        new_epsilon *= self.args.decay_rate
                    new_epsilon = max(self.epsilon, self.args.epsilon_scaler * exp_bonus / self.max_exp_bonus, new_epsilon)
                    if self.args.tb and self.T % self.args.tb_interval == 0:
                        self.log_value("Epsilon/Count", new_epsilon, step=self.T)

                action, action_info = self.select_action(state, new_epsilon)

                if not self.args.count_epsilon:
                    exp_bonus, exp_info = self.exploration_bonus(state, action)

                density = 1
                if self.args.count:
                    density = exp_info["Density"]

                if self.args.render or will_save_states:
                    debug_info = {}
                    debug_info.update(action_info)
                    debug_info.update(exp_info)
                    if self.args.render:
                        if self.args.slow_render:
                            time.sleep(0.1)
                        self.env.debug_render(debug_info)
                    if will_save_states:
                        debug_state = self.env.debug_render(debug_info, mode="rgb_array")
                        video_states.append(debug_state)

                if self.args.visitations:
                    self.visitations()

                self.visualisations()

                state_new, reward, episode_finished, env_info = self.env.step(action)
                self.T += 1
                self.episode_steps += 1

                # Action State Count stuff if available
                if "Action_Counts" in env_info:
                    self.Action_State_Counts.append(env_info["Action_Counts"])

                # If the environment terminated because it reached a limit, we do not want the agent
                # to see that transition, since it makes the env non markovian wrt state
                if "Steps_Termination" in env_info:
                    episode_finished = True
                    break

                self.episode_reward += reward

                if self.args.no_exploration_bonus:
                    # 0 out the exploration bonus
                    exp_bonus = 0

                self.episode_bonus_only_reward += exp_bonus

                self.agent.experience(state, action, reward, state_new, 1, episode_finished, exp_bonus, density)

                self.train_agent()

                state = state_new

                if not self.args.plain_print:
                    print("\x1b[K_{}_".format(self.T), end="\r")
                    if self.T % 1000 == 0:
                        self.print_time()

                if self.T - self.eval_T >= self.args.t_max // self.args.eval_interval:
                    self.eval_agent()

            self.episode += 1
            self.Episode_Rewards.append(self.episode_reward)
            self.Episode_Lengths.append(self.episode_steps)
            self.Episode_Bonus_Only_Rewards.append(self.episode_bonus_only_reward)

            self.end_of_episode()

            self.save_values()

            if will_save_states:
                self.save_video("{}/training/Training_Policy__T_{}__Ep_{}".format(self.args.log_path, self.T, self.episode), video_states)
                self.training_video_T = self.T

        self.end_of_training_save()

        print()
        print("Environment {:.2} GB".format(asizeof.asizeof(self.env) / 1024.0 ** 3))
        print("Player Positions {:.2} GB".format(asizeof.asizeof(self.Player_Positions) / 1024.0 ** 3))
        print("Visited state {:.2} GB".format(asizeof.asizeof(self.Visited_States) / 1024.0 ** 3))
        print("Trained states {:.2} GB".format(asizeof.asizeof(self.Trained_On_States) / 1024.0 ** 3))
        print("Agent {:.2} GB".format(asizeof.asizeof(self.agent) / 1024.0 ** 3))
        print("Agent Replay {:.2} GB".format(asizeof.asizeof(self.agent.replay) / 1024.0 ** 3))
        if self.args.count:
            print("Exp Model {:.2} GB".format(asizeof.asizeof(self.exp_model) / 1024.0 ** 3))

        if self.args.render:
            print("\n\nClosing render window")
            self.env.debug_render(close=True)

        print("\nEvaluating Last Agent\n")
        self.eval_agent(last=True)
        print("Last Evaluation Finished")

        # Close out the logging queue
        print("\nClosing queue")
        finished_training.value = 10
        self.log_queue.close()
        time.sleep(5)
        # print("Waiting for queue to finish")
        # p_log.join()
        p_log.join(timeout=1)

        print("\nFinished\n")
