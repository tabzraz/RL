import argparse
import gym
import datetime
import time
import os
from math import sqrt

import numpy as np
from skimage.transform import resize

import tensorflow as tf

import Exploration.CTS as CTS

from Misc.Gradients import clip_grads
from Utils.Utils import time_str
from Replay.ExpReplay_Options import ExperienceReplay_Options
from Replay.Prioritised_ExpReplay_Options import ExperienceReplay_Options as Prioritised_ExpReplay_Options
from Models.Models import get_models
import Envs

parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--t-max", type=int, default=int(1e5))
parser.add_argument("--env", type=str, default="Maze-1-v1")
parser.add_argument("--logdir", type=str, default="Logs")
parser.add_argument("--name", type=str, default="nn")
parser.add_argument("--exp-replay-size", "--xp", type=int, default=int(20e3))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--model", type=str, default="")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--exploration-steps", "--exp-steps", type=int, default=int(5e4))
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--epsilon-steps", "--eps-steps", type=int, default=int(5e4))
parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
parser.add_argument("--epsilon-finish", "--eps-finish", type=float, default=0.1)
parser.add_argument("--batch-size", "--batch", type=int, default=2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target", type=int, default=100)
parser.add_argument("--count", action="store_true", default=False)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--n-step", "--n", type=int, default=1)
parser.add_argument("--plain-print", action="store_true", default=False)
parser.add_argument("--xla", action="store_true", default=False)
parser.add_argument("--clip-value", type=float, default=5)
parser.add_argument("--no-tensorboard", "--no-tb", action="store_true", default=False)
parser.add_argument("--action-override", type=int, default=-1)
parser.add_argument("--double-q", type=bool, default=True)
parser.add_argument("--checkpoint-interval", "--ckpt", type=int, default=5e4)
parser.add_argument("--restore", type=str, default="")
parser.add_argument("--prioritized", "--pexp", action="store_true", default=False)
parser.add_argument("--bonus-after-epsilon", action="store_true", default=False)
parser.add_argument("--unroll", type=int, default=32)
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if not args.gpu:
    print("\n---DISABLING GPU---\n")
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
if args.xla:
    print("\n---USING XLA---\n")
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

args.tb = not args.no_tensorboard

if args.model == "":
    args.model = "DRQN-" + args.env

# Tensorflow sessions
sess = tf.Session(config=config)

# Seed everything
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

LOGDIR = "{}/{}_{}".format(args.logdir, args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
print("\n" + "=" * 40)
print(16 * " " + "Logdir:" + " " * 16)
print("=" * 40)
print(" Logging to:\n {}".format(LOGDIR))
print("=" * 40)

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists("{}/logs".format(LOGDIR)):
    os.makedirs("{}/logs".format(LOGDIR))
if not os.path.exists("{}/ckpts".format(LOGDIR)):
    os.makedirs("{}/ckpts".format(LOGDIR))

with open("{}/settings.txt".format(LOGDIR), "w") as f:
    f.write(str(args))

# Gym Environment
print()
env = gym.make(args.env)
args.actions = env.action_space.n
if args.action_override > 0:
    args.actions = args.action_override

# Print Settings and Hyperparameters
print("\n" + "=" * 40)
print(15 * " " + "Settings:" + " " * 15)
print("=" * 40)
for arg in vars(args):
    print(" {}: {}".format(arg, getattr(args, arg)))
print("=" * 40)
print()

# Experience Replay
replay = None
if args.prioritized:
    replay = Prioritised_ExpReplay_Options(args.exp_replay_size)
else:
    replay = ExperienceReplay_Options(args.exp_replay_size)

# DQN
print("\n" + "=" * 40)
print(16 * " " + "Models:" + " " * 16)
print("=" * 40)
dqn = get_models(args.model)(name="DQN")
print()
target_dqn = get_models(args.model)(name="Target_DQN")
print("=" * 40)

# Optimizer
optimiser = tf.train.AdamOptimizer(args.lr)

# Tensorflow Operations
with tf.name_scope("Sync_Target_DQN"):
    dqn_vars = dqn.variables
    target_dqn_vars = target_dqn.variables
    sync_vars_list = []
    for (ref, val) in zip(target_dqn_vars, dqn_vars):
        sync_vars_list.append(tf.assign(ref, val))
    sync_vars = tf.group(*sync_vars_list)
# print(dqn_vars)
# print(target_dqn_vars)

with tf.name_scope("Minimise_DQN_Loss"):
    qval_loss = dqn.q_loss
    grads_vars = optimiser.compute_gradients(qval_loss)
    clipped_grads_vars, dqn_grad_norm = clip_grads(grads_vars, args.clip_value)
    minimise_op = optimiser.apply_gradients(clipped_grads_vars)

# Checkpoints
checkpointer = tf.train.Saver(var_list=dqn.variables, max_to_keep=None)

# Tensorboard Stuff
DQN_Q_Values_Summary = dqn.qvals_summary
DQN_Loss_Summary = dqn.loss_summary
DQN_Grad_Norm_Summary = tf.summary.scalar("DQN_Gradient_Norm", dqn_grad_norm)

# Dummy variable to log scalars to tensorboard
scalar_variable = tf.Variable(name="Insert_Scalar_Here", initial_value=0.0, dtype=tf.float32, trainable=False)

if args.count:
    Count_Bonus_Summary = tf.summary.scalar("Count_Bonus", scalar_variable)

Episode_Bonus_Only_Reward_Summary = tf.summary.scalar("Episode_Bonus_Only_Rewards", scalar_variable)

Episode_Reward_Summary = tf.summary.scalar("Episode_Reward", scalar_variable)
Episode_Length_Summary = tf.summary.scalar("Episode_Length", scalar_variable)

Epsilon_Summary = tf.summary.scalar("Epsilon", scalar_variable)

training_writer = tf.summary.FileWriter("{}/tb/agent".format(LOGDIR), graph=sess.graph)


def save_scalar(value, summary, global_step, training=True):
    summary_value = sess.run(summary, {scalar_variable: value})
    save_to_tb(summary_value, global_step, training=training)


def save_to_tb(summary, global_step, training=True):
    if args.tb:
        if training:
            training_writer.add_summary(summary, global_step=global_step)


# Stuff to log
Q_Values = []
Episode_Rewards = []
Episode_Bonus_Only_Rewards = []
Episode_Lengths = []
DQN_Loss = []
Exploration_Bonus = []

# Inter-episode
Rewards = []
States = []
Actions = []
States_Next = []

Last_T_Logged = 1
Last_Ep_Logged = 1


# Variables and stuff
T = 1
episode = 1
epsilon = 1
episode_reward = 0
episode_steps = 0
episode_bonus_only_reward = 0
# RNN_State = (np.zeros(shape=(1, dqn.lstm_size)), np.zeros(shape=(1, dqn.lstm_size)))
RNN_State = None

if args.count:
    # Use half the env size
    env_size = env.shape[0] * 7
    cts_model_shape = (env_size // 2, env_size // 2)
    print("\nCTS Model has size: " + str(cts_model_shape) + "\n")
    cts_model = CTS.LocationDependentDensityModel(frame_shape=cts_model_shape, context_functor=CTS.L_shaped_context)


# Methods
def environment_specific_stuff():
    if args.env.startswith("Maze"):
        player_pos = env.player_pos
        with open("{}/logs/Player_Positions_In_Maze.txt".format(LOGDIR), "a") as file:
            file.write(str(player_pos) + "\n")


def exploration_bonus(state):
    if args.bonus_after_epsilon and T < args.epsilon_steps:
        return 0
    if args.count:
        state = resize(state, output_shape=cts_model_shape)
        rho_old = np.exp(cts_model.update(state))
        rho_new = np.exp(cts_model.log_prob(state))
        pseudo_count = (rho_old * (1 - rho_new)) / (rho_new - rho_old)
        pseudo_count = max(pseudo_count, 0)
        bonus = args.beta / sqrt(pseudo_count + 0.01)
        Exploration_Bonus.append(exp_bonus)
        save_scalar(bonus, Count_Bonus_Summary, global_step=T)
        return bonus
    return 0


def save_values():
    global Last_Ep_Logged
    if episode > Last_Ep_Logged:
        with open("{}/logs/Episode_Rewards.txt".format(LOGDIR), "ab") as file:
            np.savetxt(file, Episode_Rewards[Last_Ep_Logged - 1:], delimiter=" ", fmt="%f")

        with open("{}/logs/Episode_Lengths.txt".format(LOGDIR), "ab") as file:
            np.savetxt(file, Episode_Lengths[Last_Ep_Logged - 1:], delimiter=" ", fmt="%d")

        with open("{}/logs/Episode_Bonus_Only_Rewards.txt".format(LOGDIR), "ab") as file:
            np.savetxt(file, Episode_Bonus_Only_Rewards[Last_Ep_Logged - 1:], delimiter=" ", fmt="%d")

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
    sess.run(sync_vars)


def select_action(state):
    global RNN_State
    dqn_q_values = dqn.q_values
    dqn_inputs = dqn.inputs
    unrolls = dqn.unroll
    lstm_state = dqn.initial_lstm_state
    batch_size = dqn.batch_size
    final_states = dqn.final_states
    # print(RNN_State)
    q_values, RNN_State, q_values_summary = sess.run([dqn_q_values, final_states, DQN_Q_Values_Summary], {dqn_inputs: [state], unrolls: 1, lstm_state: RNN_State, batch_size: 1})
    q_values = q_values[0][0]
    Q_Values.append(q_values)
    save_to_tb(q_values_summary, global_step=T)
    # Epsilon Greedy
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_values)

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
    states, actions, rewards, next_states, terminals = replay.Sample_Sequence(args.batch_size, args.unroll)

    dqn_inputs = dqn.inputs
    dqn_targets = dqn.target_q
    dqn_actions = dqn.action_index
    dqn_qvals = dqn.q_values
    batch_size = dqn.batch_size
    unroll = dqn.unroll
    # dqn_state = dqn.initial_lstm_state
    # dqn_errors = dqn["Q_Error"]

    target_dqn_input = target_dqn.inputs
    target_dqn_qvals = target_dqn.q_values
    target_batch_size = target_dqn.batch_size
    target_unroll = target_dqn.unroll
    # target_dqn_state = target_dqn.initial_lstm_state

    # zero_rnn_state = (np.zeros(shape=(1, dqn.lstm_size)), np.zeros(shape=(1, dqn.lstm_size)))

    flattened_states = []
    flattened_next_states = []
    flattened_actions = []
    flattened_targets = []
    flattened_rewards = []
    flattened_terminals = []
    # sequence_lengths = []
    for state, action, reward, next_state, terminal in zip(states, actions, rewards, next_states, terminals):
        # State is a list of sequential data, likewise for the rest
        zero_state = np.zeros(shape=state[0].shape)
        # zero_action = np.zeros(shape=action[0].shape)
        length = len(state)
        # sequence_lengths.append(length)
        padded_state = state + ([zero_state] * (args.unroll - length))
        padded_action = action + ([-1] * (args.unroll - length))
        padded_next_state = next_state + ([zero_state] * (args.unroll - length))
        padded_reward = reward + ([0] * (args.unroll - length))
        padded_terminal = terminal + ([True] * (args.unroll - length))
        flattened_states += padded_state
        flattened_actions += padded_action
        flattened_next_states += padded_next_state
        flattened_rewards += padded_reward
        flattened_terminals += padded_terminal

    q_dict = {}
    q_dict[batch_size] = args.batch_size
    q_dict[target_batch_size] = args.batch_size
    q_dict[unroll] = args.unroll
    q_dict[target_unroll] = args.unroll
    q_dict[dqn_inputs] = flattened_next_states
    q_dict[target_dqn_input] = flattened_next_states
    # q_dict[dqn_state] = zero_rnn_state
    # q_dict[target_dqn_state] = zero_rnn_state
    # print(len(flattened_next_states))
    new_state_qvals, target_qvals = sess.run([dqn_qvals, target_dqn_qvals], feed_dict=q_dict)

    flattened_action_indices = []
    for action, reward, terminal, target_qval, double_qval in zip(flattened_actions, flattened_rewards, flattened_terminals, target_qvals, new_state_qvals):
        target = np.zeros(args.actions)
        if action < 0:
            # This is a padded thing, make the onehot actions 0 => no error
            flattened_targets.append(target)
            flattened_action_indices.append(np.zeros(args.actions))
        else:
            target[action] = reward
            if not terminal:
                if args.double_q:
                    target[action] += (args.gamma) * target_qval[np.argmax(double_qval)]
                else:
                    target[action] += (args.gamma) * np.max(target_qval)
            flattened_targets.append(target)
            action_onehot = np.zeros(args.actions)
            action_onehot[action] = 1
            flattened_action_indices.append(action_onehot)

    # seq_lens = dqn.sequence_lengths
    feed_dict = {}
    feed_dict[batch_size] = args.batch_size
    feed_dict[unroll] = args.unroll
    feed_dict[dqn_inputs] = flattened_states
    feed_dict[dqn_targets] = flattened_targets
    feed_dict[dqn_actions] = flattened_action_indices
    # print(len(flattened_targets))
    # _ = sess.run([minimise_op], feed_dict=feed_dict)
    _, loss_summary, norm_summary = sess.run([minimise_op, DQN_Loss_Summary, DQN_Grad_Norm_Summary], feed_dict=feed_dict)

    # # Create targets from the batch
    # # old_states = list(map(lambda tups: tups[0], batch))
    # # new_states = list(map(lambda tups: tups[3], batch))
    # new_state_qvals, target_qvals = sess.run([dqn_qvals, target_dqn_qvals], feed_dict={target_dqn_input: next_states, dqn_inputs: next_states})
    # q_targets = []
    # actions = []

    # for batch_item, target_qval, double_qvals in zip(batch, target_qvals, new_state_qvals):
    #     st, at, rt, snew, steps, terminal = batch_item
    #     target = np.zeros(args.actions)
    #     target[at] = rt
    #     if not terminal:
    #         if args.double_q:
    #             target[at] += (args.gamma ** steps) * target_qval[np.argmax(double_qvals)]
    #         else:
    #             target[at] += (args.gamma ** steps) * np.max(target_qval)
    #     q_targets.append(target)
    #     action_onehot = np.zeros(args.actions)
    #     action_onehot[at] = 1
    #     actions.append(action_onehot)

    # _, q_errors, loss_summary, norm_summary = sess.run([minimise_op, dqn_errors, DQN_Loss_Summary, DQN_Grad_Norm_Summary], feed_dict={dqn_inputs: old_states, dqn_targets: q_targets, dqn_actions: actions})

    # if args.prioritized:
    #     td_errors = list(map(lambda x: np.max(x), q_errors))
    #     replay.Update_Indices(indices, td_errors)

    save_to_tb(loss_summary, global_step=T)
    save_to_tb(norm_summary, global_step=T)


def start_of_episode_tb():
    save_scalar(epsilon, Epsilon_Summary, T)


def end_of_episode_tb():
    save_scalar(episode_bonus_only_reward, Episode_Bonus_Only_Reward_Summary, T)
    save_scalar(episode_reward, Episode_Reward_Summary, T)
    save_scalar(episode_steps, Episode_Length_Summary, T)


def save_checkpoint():
    checkpointer.save(sess=sess, save_path="{}/ckpts/dqn".format(LOGDIR), global_step=T)


######################
# Training procedure #
######################

explore()

sess.run(tf.global_variables_initializer())

if args.restore != "":
    print("\n--RESTORING--\nFrom: {}".format(args.restore))
    checkpointer.restore(sess, save_path=args.restore)

sync_target_network()

start_time = time.time()

print("Training.\n\n\n")

while T < args.t_max:

    state = env.reset()
    if args.render:
        env.render()
    episode_finished = False
    episode_reward = 0
    episode_steps = 0
    episode_bonus_only_reward = 0
    exp_bonus = 0
    RNN_State = (np.zeros(shape=(1, dqn.lstm_size)), np.zeros(shape=(1, dqn.lstm_size)))
    # RNN_State = dqn.zero_state(args.batch_size)

    epsilon = epsilon_schedule()

    start_of_episode_tb()

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

        exp_bonus = exploration_bonus(state)

        episode_bonus_only_reward += exp_bonus
        reward += exp_bonus

        episode_steps += 1
        T += 1

        Rewards.append(reward)
        States.append(state)
        States_Next.append(state_new)
        Actions.append(action)

        replay.Add_Exp(state, action, reward, state_new, 1, episode_finished)

        train_agent()

        state = state_new

        if T % args.target == 0:
            sync_target_network()

        if T % args.checkpoint_interval == 0:
            save_checkpoint()

        environment_specific_stuff()

        if not args.plain_print:
            print("\x1b[K" + "." * ((episode_steps // 20) % 40), end="\r")

    episode += 1
    Episode_Rewards.append(episode_reward)
    Episode_Lengths.append(episode_steps)
    Episode_Bonus_Only_Rewards.append(episode_bonus_only_reward)

    end_of_episode_tb()

    # TODO: Do something with these or get rid of them
    Rewards = []
    States = []
    States_Next = []
    Actions = []

    save_values()

# Housekeeping
checkpointer.save(sess=sess, save_path="{}/ckpts/dqn-final".format(LOGDIR), global_step=T)

if args.render:
    env.render(close=True)
# Keep the tensorflow session open incase we run this in ipython
# sess.close()

print("\nFinished\n")
