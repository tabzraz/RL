from __future__ import division, print_function
import numpy as np
import gym
import tensorflow as tf
import time
import datetime
import os
import json
from collections import deque
from math import sqrt, log
import dill
from Misc.Gradients import clip_grads
from Replay.ExpReplay import ExperienceReplay
import Exploration.CTS as CTS
# Todo: Make this friendlier
from Models.DQN_Maze import model
from Models.VIME_Maze import model as exploration_model
# import gym_minecraft
import Envs

# Things still to implement

flags = tf.app.flags
flags.DEFINE_string("env", "Maze-2-v1", "Environment name for OpenAI gym")
flags.DEFINE_string("logdir", "", "Directory to put logs (including tensorboard logs)")
flags.DEFINE_string("name", "nn", "The name of the model")
flags.DEFINE_float("lr", 0.0001, "Initial Learning Rate")
flags.DEFINE_float("vime_lr", 0.0001, "Initial Learning Rate for VIME model")
flags.DEFINE_float("gamma", 0.99, "Gamma, the discount rate for future rewards")
flags.DEFINE_integer("t_max", int(2e5), "Number of frames to act for")
flags.DEFINE_integer("episodes", 100, "Number of episodes to act for")
flags.DEFINE_integer("action_override", 0, "Overrides the number of actions provided by the environment")
flags.DEFINE_float("grad_clip", 10, "Clips gradients by their norm")
flags.DEFINE_integer("seed", 0, "Seed for numpy and tf")
flags.DEFINE_integer("ckpt_interval", 5e4, "How often to save the global model")
flags.DEFINE_integer("xp", int(5e4), "Size of the experience replay")
flags.DEFINE_float("epsilon_start", 1.0, "Value of epsilon to start with")
flags.DEFINE_float("epsilon_finish", 0.1, "Final value of epsilon to anneal to")
flags.DEFINE_integer("epsilon_steps", int(10e4), "Number of steps to anneal epsilon for")
flags.DEFINE_integer("target", 100, "After how many steps to update the target network")
flags.DEFINE_boolean("double", True, "Double DQN or not")
flags.DEFINE_integer("batch", 64, "Minibatch size")
flags.DEFINE_integer("summary", 5, "After how many steps to log summary info")
flags.DEFINE_integer("exp_steps", int(1e4), "Number of steps to randomly explore for")
flags.DEFINE_boolean("render", False, "Render environment or not")
flags.DEFINE_string("ckpt", "", "Model checkpoint to restore")
flags.DEFINE_boolean("vime", False, "Whether to add VIME style exploration bonuses")
flags.DEFINE_integer("posterior_iters", 1, "Number of times to run gradient descent for calculating new posterior from old posterior")
flags.DEFINE_float("eta", 0.1, "How much to scale the VIME rewards")
flags.DEFINE_integer("vime_iters", 50, "Number of iterations to optimise the variational baseline for VIME")
flags.DEFINE_integer("vime_batch", 32, "Size of minibatch for VIME")
flags.DEFINE_boolean("rnd", False, "Random Agent")
flags.DEFINE_boolean("pseudo", False, "PseudoCount bonuses or not")
flags.DEFINE_integer("n", 10, "Number of steps for n-step Q-Learning")
flags.DEFINE_float("beta", 0.10, "Beta for pseudocounts")

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
VIME_ITERS = FLAGS.vime_iters
NAME = FLAGS.name
EPISODES = FLAGS.episodes
T_MAX = FLAGS.t_max
EPSILON_START = FLAGS.epsilon_start
EPSILON_FINISH = FLAGS.epsilon_finish
EPSILON_STEPS = FLAGS.epsilon_steps
XP_SIZE = FLAGS.xp
GAMMA = FLAGS.gamma
BATCH_SIZE = FLAGS.batch
TARGET_UPDATE = FLAGS.target
SUMMARY_UPDATE = FLAGS.summary
CLIP_VALUE = FLAGS.grad_clip
VIME = FLAGS.vime
VIME_POSTERIOR_ITERS = FLAGS.posterior_iters
RANDOM_AGENT = FLAGS.rnd
CHECKPOINT_INTERVAL = FLAGS.ckpt_interval
N_STEPS = FLAGS.n
PSEDUOCOUNT = FLAGS.pseudo
BETA = FLAGS.beta
if FLAGS.ckpt != "":
    RESTORE = True
    CHECKPOINT = FLAGS.ckpt
else:
    RESTORE = False
EXPLORATION_STEPS = FLAGS.exp_steps
LOGNAME = "Logs"
if FLAGS.logdir != "":
    LOGNAME = FLAGS.logdir
LOGDIR = "{}/{}_{}".format(LOGNAME, NAME, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
if not os.path.exists("{}/ckpts/".format(LOGDIR)):
    os.makedirs("{}/ckpts".format(LOGDIR))

# Print all the hyperparams
hyperparams = FLAGS.__dict__["__flags"]
with open("{}/info.json".format(LOGDIR), "w") as fp:
    json.dump(hyperparams, fp)


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
        # Seed numpy and tensorflow
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        # tflearn.config.init_graph(seed=SEED)

        sess = tf.Session(config=config)
    # with tf.Session(config=config) as sess:

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

        with tf.name_scope("Start_State"):
            start_qvals = []
            for i in range(ACTIONS):
                start_qvals.append(tf.summary.scalar("Start State Action {} QValue".format(i), dqn_qvals[0, i]))
            dqn_start_state_qvals_summary = tf.summary.merge(start_qvals)

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
            vime_grad_norm_summary = tf.summary.scalar("Vime_Gradients_Norm", vime_grad_norm)
            vime_minimise_op = vime_optimizer.apply_gradients(vime_clipped)

            vime_posterior_grads = vime_optimizer.compute_gradients(vime_posterior_loss)
            vime_posterior_clipped, vime_posterior_grad_norm = clip_grads(vime_posterior_grads, CLIP_VALUE)
            vime_posterior_minimise_op = vime_optimizer.apply_gradients(vime_posterior_clipped)

        if PSEDUOCOUNT:
            # We will cheat a bit and use tabular counting to begin with
            # state_counter = {}
            # Cts model
            cts_model = CTS.LocationDependentDensityModel(frame_shape=(env.shape[0] * 7, env.shape[0] * 7, 1), context_functor=CTS.L_shaped_context)
            count_bonus = tf.placeholder(tf.float32)
            count_bonus_summary = tf.summary.scalar("Pseudocount_Bonus", count_bonus)
            count_episode = tf.placeholder(tf.float32)
            count_episode_summary = tf.summary.scalar("Count Episode Reward", count_episode)

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
        reward_episode_summary = tf.summary.scalar("Per Episode Reward", episode_reward)
        tf_epsilon = tf.placeholder(tf.float32)
        epsilon_summary = tf.summary.scalar("Epsilon", tf_epsilon)
        episode_length = tf.placeholder(tf.int32)
        length_summary = tf.summary.scalar("Episode Length", episode_length)

        if VIME:
            vime_episode_reward = tf.placeholder(tf.float32)
            vime_episode_reward_summary = tf.summary.scalar("Vime Episode Reward", vime_episode_reward)
            vime_kldiv_summary = tf.summary.scalar("KL", vime_kldiv)
            vime_reward = tf.placeholder(tf.float32)
            vime_rewards_summary = tf.summary.scalar("Vime Rewards", vime_reward)
            vime_loss_summary = tf.summary.scalar("Vime Loss", vime_loss)
            vime_posterior_loss_summary = tf.summary.scalar("Vime Posterior Loss", vime_posterior_loss)
            vime_kls = deque()
            # vime_kls.append(1.0)

        sess.run(tf.global_variables_initializer())
        sess.run(sync_vars)
        #Testing init
        # for i in range(100):
            # print(np.random.random())
        # weights = sess.run(dqn_vars)
        # print(weights)

        saver = tf.train.Saver(max_to_keep=None)

        if RESTORE:
            print("\n--RESTORING--\nFrom: {}\n".format(CHECKPOINT))
            saver.restore(sess, save_path=CHECKPOINT)

        writer = tf.summary.FileWriter("{}/tb_logs/dqn_agent".format(LOGDIR), graph=sess.graph)

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

        # if PSEDUOCOUNT:
        #     batch = replay.Sample(100)
        #     states = list(map(lambda tups: tups[0], batch))
        #     log_prob = 0
        #     for s in states:
        #         log_prob += cts_model.update(s)
        #     print('Loss (in bytes per frame): {:.2f}'.format(-log_prob / log(2) / 8 / 100))

        print("Exploratory phase finished, starting learning")
        start_time = time.time()

        while T < T_MAX:

            frames = 0

            s_t = env.reset()
            if RENDER:
                env.render()
            episode_finished = False

            epsilon = EPSILON_FINISH + (EPSILON_START - EPSILON_FINISH) * max(((EPSILON_STEPS - T) / EPSILON_STEPS), 0)

            time_elapsed = time.time() - start_time
            time_left = time_elapsed * (T_MAX - T) / T
            # Just in case, 100 days is the upper limit
            time_left = min(time_left, 60 * 60 * 24 * 100)

            print("Ep: {:,}, T: {:,}/{:,}, Epsilon: {:.2f}, Elapsed: {}, Left: {}".format(episode, T, T_MAX, epsilon, time_str(time_elapsed), time_str(time_left)), " " * 10, end="\r")

            _, start_qvals_summary = sess.run([dqn_qvals, dqn_start_state_qvals_summary], feed_dict={dqn_inputs: [s_t]})
            writer.add_summary(start_qvals_summary, global_step=episode)
            # Check Q Values for start state for testing
            # test_qval_summaries = sess.run(dqn_summary_qvals, feed_dict={dqn_inputs: [test_state]})
            # test_writer.add_summary(test_qval_summaries, global_step=T)
            # ----

            if VIME:
                sess.run(vime_set_baseline)

            ep_reward = 0
            if VIME:
                vime_ep_reward = 0
            if PSEDUOCOUNT:
                count_ep_reward = 0
            while not episode_finished:

                # Random agent
                if RANDOM_AGENT:
                    action = env.action_space.sample()
                    _, r_t, episode_finished, _ = env.step(action)
                    ep_reward += r_t
                    T += 1
                    continue

                q_vals, qvals_summary = sess.run([dqn_qvals, dqn_summary_qvals], feed_dict={dqn_inputs: [s_t]})
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_vals[0, :])

                s_new, r_t, episode_finished, info_dict = env.step(action)
                if RENDER:
                    env.render()
                if "Steps_Termination" not in info_dict:
                    # We have taken too long in the environment, so the episode is ending
                    # We do NOT want to show this transition to the agent
                    # print("Step Termination seen")
                    ep_reward += r_t

                    if VIME:
                        sess.run(vime_set_variational_params)
                        # Compute posterior
                        one_hot_action = np.zeros(ACTIONS)
                        one_hot_action[action] = 1
                        # TODO: Use Hessian
                        states = [s_t for _ in range(VIME_POSTERIOR_ITERS)]
                        actions = [one_hot_action for _ in range(VIME_POSTERIOR_ITERS)]
                        targets = [s_new for _ in range(VIME_POSTERIOR_ITERS)]
                        _, vime_posterior_loss_summary_value = sess.run([vime_posterior_minimise_op, vime_posterior_loss_summary], feed_dict={vime_input: states, vime_action: actions, vime_target: targets, vime_kl_scaling: VIME_POSTERIOR_ITERS * 1.0})
                        kldiv = sess.run(vime_kldiv)
                        reward_to_give = kldiv
                        vime_kls.append(reward_to_give)
                        if len(vime_kls) > 1000:
                            vime_kls.popleft()
                        reward_to_give *= ETA
                        reward_to_give /= max(np.median(vime_kls), 1e-4)
                        kldiv_summary, vime_reward_summary = sess.run([vime_kldiv_summary, vime_rewards_summary], feed_dict={vime_reward: reward_to_give})
                        r_t += reward_to_give
                        vime_ep_reward += r_t

                    if PSEDUOCOUNT:
                        # Super hard-coded for the maze
                        # count_state = s_t
                        # player_pos = np.argwhere(count_state > 0.9)[0]
                        # goals_left = (abs(count_state - 0.6666) < 0.1).sum()
                        # state_str = (player_pos[0], player_pos[1], goals_left)
                        # # print(state_str)
                        # if state_str not in state_counter:
                        #     state_counter[state_str] = 1
                        # else:
                        #     state_counter[state_str] += 1
                        # bonus = BETA / sqrt(state_counter[state_str])

                        rho_old = np.exp(cts_model.update(s_t))
                        # cts_model.update(s_t)
                        rho_new = np.exp(cts_model.log_prob(s_t))
                        # print(rho_old, " ,", rho_new)
                        pseudo_count = (rho_old * (1 - rho_new)) / (rho_new - rho_old)
                        pseudo_count = max(pseudo_count, 0)  # Hack
                        # print(pseudo_count)
                        bonus = BETA / sqrt(pseudo_count + 0.01)

                        r_t += bonus
                        count_ep_reward += r_t
                        count_bonus_summary_val = sess.run(count_bonus_summary, {count_bonus: bonus})

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

                    if PSEDUOCOUNT:
                        writer.add_summary(count_bonus_summary_val, global_step=T)

                if T % CHECKPOINT_INTERVAL == 0:
                    saver.save(sess=sess, save_path="{}/ckpts/vars-{}.ckpt".format(LOGDIR, T))

            r_summary, r_eps_summary, e_summary, l_summary = sess.run([reward_summary, reward_episode_summary, epsilon_summary, length_summary], feed_dict={episode_length: frames, tf_epsilon: epsilon, episode_reward: ep_reward})
            writer.add_summary(r_summary, global_step=T)
            writer.add_summary(r_eps_summary, global_step=episode)
            writer.add_summary(e_summary, global_step=T)
            writer.add_summary(l_summary, global_step=T)
            if VIME:
                vr_summary = sess.run(vime_episode_reward_summary, feed_dict={vime_episode_reward: vime_ep_reward})
                writer.add_summary(vr_summary, global_step=T)
            if PSEDUOCOUNT:
                c_summary = sess.run(count_episode_summary, feed_dict={count_episode: count_ep_reward})
                writer.add_summary(c_summary, global_step=T)

            episode += 1

            # if PSEDUOCOUNT:
                # # TODO: Log this if possible
                # print("Size of state counter: {}____".format(len(state_counter)))

            if VIME:
                sess.run(vime_revert_to_baseline)
                for _ in range(VIME_ITERS):
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
        print("\nRunning final episode evaluation")
        eval_writer = tf.summary.FileWriter("{}/tb_logs/dqn_eval".format(LOGDIR), graph=sess.graph)
        s = env.reset()
        steps = 0
        cumulative_reward = 0
        reward_tensor = tf.placeholder(tf.float32)
        c_reward_summay = tf.summary.scalar("Reward Over Episode", reward_tensor)
        terminated = False
        states_to_save = [s]
        while not terminated:
            q_vals, qvals_summary = sess.run([dqn_qvals, dqn_summary_qvals], feed_dict={dqn_inputs: [s]})
            if np.random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(q_vals[0, :])
            sn, r, terminated, _ = env.step(a)
            states_to_save.append(sn)
            s = sn
            # if sn == states_to_save[-2]:
                # Same state again => repeat behaviour
                # terminated = True
            steps += 1
            cumulative_reward += r
            c_r_s_v = sess.run(c_reward_summay, {reward_tensor: cumulative_reward})
            eval_writer.add_summary(qvals_summary, global_step=steps)
            eval_writer.add_summary(c_r_s_v, global_step=steps)

        states_to_save_tensor = tf.placeholder(tf.float32, shape=[len(states_to_save), env.shape[0] * 7, env.shape[0] * 7, 1])
        states_summary = tf.summary.image("States", states_to_save_tensor, max_outputs=len(states_to_save))
        states_summary_value = sess.run(states_summary, {states_to_save_tensor: states_to_save})
        eval_writer.add_summary(states_summary_value)

        if RENDER:
            env.render(close=True)

        # if PSEDUOCOUNT:
            # dill.dump(cts_model, open("{}/cts_model".format(LOGDIR), "w"))
        # Save the final model
        saver.save(sess=sess, save_path="{}/ckpts/vars-{}-FINAL.ckpt".format(LOGDIR, T))

        print("\nFinished")
