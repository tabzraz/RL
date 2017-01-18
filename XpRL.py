from __future__ import division, print_function
import numpy as np
import gym
import tensorflow as tf
import time
import datetime
import os
from tqdm import tqdm
from Misc.Gradients import clip_grads
from Replay.ExpReplay import ExperienceReplay
# Todo: Make this friendlier
from Models.DQN_Maze import model
from Models.VIME_Maze import model as exploration_model
# import gym_minecraft
import Envs

# Things still to implement

flags = tf.app.flags
flags.DEFINE_string("env", "Maze-4-v0", "Environment name for OpenAI gym")
flags.DEFINE_string("logdir", "", "Directory to put logs (including tensorboard logs)")
flags.DEFINE_string("name", "nn", "The name of the model")
flags.DEFINE_float("lr", 0.001, "Initial Learning Rate")
flags.DEFINE_float("vime_lr", 0.001, "Initial Learning Rate for VIME model")
flags.DEFINE_float("gamma", 0.99, "Gamma, the discount rate for future rewards")
flags.DEFINE_integer("t_max", int(2e5), "Number of frames to act for")
flags.DEFINE_integer("episodes", 100, "Number of episodes to act for")
flags.DEFINE_integer("action_override", 0, "Overrides the number of actions provided by the environment")
flags.DEFINE_float("grad_clip", 50, "Clips gradients by their norm")
flags.DEFINE_integer("seed", 0, "Seed for numpy and tf")
flags.DEFINE_integer("ckpt_interval", 2e4, "How often to save the global model")
flags.DEFINE_integer("xp", int(1e4), "Size of the experience replay")
flags.DEFINE_float("epsilon_start", 1.0, "Value of epsilon to start with")
flags.DEFINE_float("epsilon_finish", 0.1, "Final value of epsilon to anneal to")
flags.DEFINE_integer("target", 500, "After how many steps to update the target network")
flags.DEFINE_boolean("double", True, "Double DQN or not")
flags.DEFINE_integer("batch", 32, "Minibatch size")
flags.DEFINE_integer("summary", 10, "After how many steps to log summary info")
flags.DEFINE_integer("exp_steps", int(1e4), "Number of steps to randomly explore for")
flags.DEFINE_boolean("render", False, "Render environment or not")
flags.DEFINE_string("ckpt", "", "Model checkpoint to restore")
flags.DEFINE_boolean("vime", False, "Whether to add VIME style exploration bonuses")
flags.DEFINE_integer("posterior_iters", 1, "Number of times to run gradient descent for calculating new posterior from old posterior")
flags.DEFINE_float("eta", 0.0001, "How much to scale the VIME rewards")
flags.DEFINE_integer("vime_epochs", 100, "Number of epochs to optimise the variational baseline for VIME")
flags.DEFINE_integer("vime_batch", 32, "Size of minibatch for VIME")

FLAGS = flags.FLAGS
ENV_NAME = FLAGS.env
RENDER = FLAGS.render
env = gym.make(ENV_NAME)

if FLAGS.action_override > 0:
    ACTIONS = FLAGS.action_override
else:
    ACTIONS = env.action_space.n
DOUBLE_DQN = FLAGS.double
SEED = FLAGS.seed
LR = FLAGS.lr
VIME_LR = FLAGS.vime_lr
ETA = FLAGS.eta
VIME_BATCH_SIZE = FLAGS.vime_batch
VIME_EPOCHS = FLAGS.vime_epochs
NAME = FLAGS.name
EPISODES = FLAGS.episodes
T_MAX = FLAGS.t_max
EPSILON_START = FLAGS.epsilon_start
EPSILON_FINISH = FLAGS.epsilon_finish
XP_SIZE = FLAGS.xp
GAMMA = FLAGS.gamma
BATCH_SIZE = FLAGS.batch
TARGET_UPDATE = FLAGS.target
SUMMARY_UPDATE = FLAGS.summary
CLIP_VALUE = FLAGS.grad_clip
VIME = FLAGS.vime
VIME_POSTERIOR_ITERS = FLAGS.posterior_iters
CHECKPOINT_INTERVAL = FLAGS.ckpt_interval
if FLAGS.ckpt != "":
    RESTORE = True
    CHECKPOINT = FLAGS.ckpt
else:
    RESTORE = False
EXPLORATION_STEPS = FLAGS.exp_steps
if FLAGS.logdir != "":
    LOGDIR = FLAGS.logdir
else:
    LOGDIR = "Logs/{}_{}".format(NAME, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

if not os.path.exists("{}/ckpts/".format(LOGDIR)):
    os.makedirs("{}/ckpts".format(LOGDIR))


def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string


print("\n--------Info--------")
print("Logdir:", LOGDIR)
print("T: {:,}".format(T_MAX))
print("Actions:", ACTIONS)
print("Gamma", GAMMA)
print("Learning Rate:", LR)
print("Batch Size:", BATCH_SIZE)
print("VIME bonuses:", VIME)
print("--------------------\n")

# TODO: Prioritized experience replay
replay = ExperienceReplay(XP_SIZE)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Graph().as_default():
    with tf.Session(config=config) as sess:

        # Seed numpy and tensorflow
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        test_state = env.reset()

        print("\n-------Models-------")
        # DQN
        dqn = model(name="DQN", actions=ACTIONS, size=env.shape[0])
        target_dqn = model(name="Target_Network", actions=ACTIONS, size=env.shape[0])

        dqn_inputs = dqn["Input"]
        target_dqn_input = target_dqn["Input"]
        dqn_qvals = dqn["Q_Values"]
        target_dqn_qvals = target_dqn["Q_Values"]
        dqn_vars = dqn["Variables"]
        target_dqn_vars = target_dqn["Variables"]
        dqn_targets = dqn["Targets"]
        dqn_actions = dqn["Actions"]
        dqn_summary_loss = dqn["Loss_Summary"]
        dqn_summary_qvals = dqn["QVals_Summary"]

        with tf.name_scope("Sync_Target_DQN"):
            sync_vars_list = []
            for (ref, val) in zip(target_dqn_vars, dqn_vars):
                sync_vars_list.append(tf.assign(ref, val))
            sync_vars = tf.group(*sync_vars_list)

        # VIME
        if VIME:
            vime_net = exploration_model(name="VIME_Model", size=env.shape[0], actions=ACTIONS)
            vime_input = vime_net["Input"]
            vime_action = vime_net["Action"]
            vime_target = vime_net["Target"]
            vime_loss = vime_net["Loss"]
            vime_posterior_loss = vime_net["Posterior_Loss"]
            vime_set_variational_params = vime_net["Set_Params"]
            vime_kldiv = vime_net["KL_Div"]
            vime_set_baseline = vime_net["Set_Baseline"]
            vime_revert_to_baseline = vime_net["Revert_Baseline"]
            vime_kl_scaling = vime_net["KL_Scaling"]

            vime_optimizer = tf.train.AdamOptimizer(VIME_LR)

            vime_grad_vars = vime_optimizer.compute_gradients(vime_loss)
            vime_clipped, vime_grad_norm = clip_grads(vime_grad_vars, CLIP_VALUE)
            vime_grad_norm_summary = tf.summary.scalar("VIME Gradients Norm", vime_grad_norm)
            vime_minimise_op = vime_optimizer.apply_gradients(vime_clipped)

            vime_posterior_grads = vime_optimizer.compute_gradients(vime_posterior_loss)
            vime_posterior_clipped, vime_posterior_grad_norm = clip_grads(vime_posterior_grads, CLIP_VALUE)
            vime_posterior_minimise_op = vime_optimizer.apply_gradients(vime_posterior_clipped)

        T = 1
        episode = 1
        qval_loss = dqn["Q_Loss"]

        optimiser = tf.train.AdamOptimizer(LR)
        grads_vars = optimiser.compute_gradients(qval_loss)
        clipped_grads_vars, dqn_grad_norm = clip_grads(grads_vars, CLIP_VALUE)
        dqn_grad_norm_summary = tf.summary.scalar("DQN Gradients Norm", dqn_grad_norm)
        minimise_op = optimiser.apply_gradients(clipped_grads_vars)

        episode_reward = tf.placeholder(tf.float32)
        reward_summary = tf.summary.scalar("Episode Reward", episode_reward)
        tf_epsilon = tf.placeholder(tf.float32)
        epsilon_summary = tf.summary.scalar("Epsilon", tf_epsilon)
        episode_length = tf.placeholder(tf.int32)
        length_summary = tf.summary.scalar("Episode Length", episode_length)

        if VIME:
            vime_episode_reward = tf.placeholder(tf.float32)
            vime_episode_reward_summary = tf.summary.scalar("Episode Reward with VIME", vime_episode_reward)
            vime_kldiv_summary = tf.summary.scalar("KL", vime_kldiv)
            vime_rewards_summary = tf.summary.scalar("Vime Rewards", vime_kldiv * ETA)
            vime_loss_summary = tf.summary.scalar("Vime Loss", vime_loss)
            vime_posterior_loss_summary = tf.summary.scalar("Vime Posterior Loss", vime_posterior_loss)

        sess.run(tf.global_variables_initializer())
        sess.run(sync_vars)

        saver = tf.train.Saver(max_to_keep=None)

        if RESTORE:
            print("\n--RESTORING--\nFrom: {}\n".format(CHECKPOINT))
            saver.restore(sess, save_path=CHECKPOINT)

        writer = tf.summary.FileWriter("{}/tb_logs/dqn_agent".format(LOGDIR), graph=sess.graph)

        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        print("Exploratory phase for {} steps".format(EXPLORATION_STEPS))
        e_steps = 0
        while e_steps < EXPLORATION_STEPS:
            s = env.reset()
            terminated = False
            while not terminated:
                print(e_steps, end="\r")
                a = env.action_space.sample()
                sn, r, terminated, _ = env.step(a)
                replay.Add_Exp(s, a, r, sn, terminated)
                s = sn
                e_steps += 1

        print("Exploratory phase finished, starting learning")
        start_time = time.time()

        while T < T_MAX:

            frames = 0

            s_t = env.reset()
            if RENDER:
                env.render()
            episode_finished = False

            epsilon = EPSILON_FINISH + (EPSILON_START - EPSILON_FINISH) * ((T_MAX - T) / T_MAX)

            time_elapsed = time.time() - start_time
            time_left = time_elapsed * (T_MAX - T) / T
            # Just in case, 100 days is the upper limit
            time_left = min(time_left, 60 * 60 * 24 * 100)

            print("Ep: {:,}, T: {:,}/{:,}, Epsilon: {:.2f}, Elapsed: {}, Left: {}".format(episode, T, T_MAX, epsilon, time_str(time_elapsed), time_str(time_left)), " " * 10, end="\r")

            # Check Q Values for start state for testing
            # test_qval_summaries = sess.run(dqn_summary_qvals, feed_dict={dqn_inputs: [test_state]})
            # test_writer.add_summary(test_qval_summaries, global_step=T)
            # ----

            if VIME:
                sess.run(vime_set_baseline)

            ep_reward = 0
            if VIME:
                vime_ep_reward = 0
            while not episode_finished:
                # env.render()
                # TODO: Exploratory phase
                q_vals, qvals_summary = sess.run([dqn_qvals, dqn_summary_qvals], feed_dict={dqn_inputs: [s_t]})
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_vals[0, :])

                s_new, r_t, episode_finished, _ = env.step(action)
                if RENDER:
                    env.render()
                ep_reward += r_t

                if VIME:
                    sess.run(vime_set_variational_params)
                    # Compute posterior
                    one_hot_action = np.zeros(ACTIONS)
                    one_hot_action[action] = 1
                    for _ in range(VIME_POSTERIOR_ITERS):
                        _, vime_posterior_loss_summary_value = sess.run([vime_posterior_minimise_op, vime_posterior_loss_summary], feed_dict={vime_input: [s_t], vime_action: [one_hot_action], vime_target: [s_new], vime_kl_scaling: 1.0})
                    kldiv, kldiv_summary, vime_reward_summary = sess.run([vime_kldiv, vime_kldiv_summary, vime_rewards_summary])
                    r_t += ETA * kldiv
                    vime_ep_reward += r_t

                replay.Add_Exp(s_t, action, r_t, s_new, episode_finished)
                s_t = s_new

                batch = replay.Sample(BATCH_SIZE)
                # Create targets from the batch
                old_states = list(map(lambda tups: tups[0], batch))
                new_states = list(map(lambda tups: tups[3], batch))
                new_state_qvals, target_qvals = sess.run([dqn_qvals, target_dqn_qvals], feed_dict={target_dqn_input: new_states, dqn_inputs: new_states})
                q_targets = []
                actions = []
                for batch_item, target_qval, double_qvals in zip(batch, target_qvals, new_state_qvals):
                    st, at, rt, snew, terminal = batch_item
                    # Reward clipping
                    # rt = np.clip(rt, -1, 1)
                    target = np.zeros(ACTIONS)
                    target[at] = rt
                    if not terminal:
                        if DOUBLE_DQN:
                            target[at] += GAMMA * target_qval[np.argmax(double_qvals)]
                        else:
                            target[at] += GAMMA * np.max(target_qval)
                    q_targets.append(target)
                    action_onehot = np.zeros(ACTIONS)
                    action_onehot[at] = 1
                    actions.append(action_onehot)

                # Minimise
                loss_summary, dqn_norm_summary, _ = sess.run([dqn_summary_loss, dqn_grad_norm_summary, minimise_op], feed_dict={dqn_inputs: old_states, dqn_targets: q_targets, dqn_actions: actions})
                frames += 1
                T += 1

                if T % TARGET_UPDATE == 0:
                    # print("Before", sess.run(target_dqn_qvals, feed_dict={target_dqn_input: test_state[np.newaxis, :]}))
                    sess.run(sync_vars)
                    # print("After", sess.run(target_dqn_qvals, feed_dict={target_dqn_input: test_state[np.newaxis, :]}))

                if T % SUMMARY_UPDATE == 0:
                    writer.add_summary(loss_summary, global_step=T)
                    writer.add_summary(qvals_summary, global_step=T)
                    writer.add_summary(dqn_norm_summary, global_step=T)

                    if VIME:
                        writer.add_summary(vime_posterior_loss_summary_value, global_step=T)
                        writer.add_summary(kldiv_summary, global_step=T)
                        writer.add_summary(vime_reward_summary, global_step=T)

                if T % CHECKPOINT_INTERVAL == 0:
                    saver.save(sess=sess, save_path="{}/ckpts/vars-{}.ckpt".format(LOGDIR, T))

            r_summary, e_summary, l_summary = sess.run([reward_summary, epsilon_summary, length_summary], feed_dict={episode_length: frames, tf_epsilon: epsilon, episode_reward: ep_reward})
            writer.add_summary(r_summary, global_step=T)
            writer.add_summary(e_summary, global_step=T)
            writer.add_summary(l_summary, global_step=T)
            if VIME:
                vr_summary = sess.run(vime_episode_reward_summary, feed_dict={vime_episode_reward: vime_ep_reward})
                writer.add_summary(vr_summary, global_step=T)

            episode += 1

            if VIME:
                sess.run(vime_revert_to_baseline)
                for _ in range(VIME_EPOCHS // VIME_BATCH_SIZE):
                    batch = replay.Sample(VIME_BATCH_SIZE)
                    old_states = list(map(lambda tups: tups[0], batch))
                    new_states = list(map(lambda tups: tups[3], batch))
                    action_indices = list(map(lambda tups: tups[1], batch))
                    actions = np.zeros((VIME_BATCH_SIZE, ACTIONS))
                    np.put(actions, action_indices, 1)
                    _, vime_loss_summary_val, vime_norm_summary_val = sess.run([vime_minimise_op, vime_loss_summary, vime_grad_norm_summary], feed_dict={vime_input: old_states, vime_action: actions, vime_target: new_states, vime_kl_scaling: XP_SIZE / VIME_BATCH_SIZE})

                    writer.add_summary(vime_loss_summary_val, global_step=T)
                    writer.add_summary(vime_norm_summary_val, global_step=T)


            # TODO: Evaluation episodes with just greedy policy, track qvalues over the episode
        if RENDER:
            env.render(close=True)

        # Save the final model
        saver.save(sess=sess, save_path="{}/ckpts/vars-{}-FINAL.ckpt".format(LOGDIR, T))

        print("\nFinished")
