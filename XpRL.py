import numpy as np
import gym
import tensorflow as tf
import time
import datetime
import os
from tqdm import tqdm
from Replay.ExpReplay import ExperienceReplay
from Models.DQN_FrozenLake import model
import Envs

flags = tf.app.flags
flags.DEFINE_string("env", "FrozenLake-v0", "Environment name for OpenAI gym")
flags.DEFINE_string("logdir", "", "Directory to put logs (including tensorboard logs)")
flags.DEFINE_string("name", "nn", "The name of the model")
flags.DEFINE_float("learning_rate", 0.0001, "Initial Learning Rate")
flags.DEFINE_float("gamma", 0.95, "Gamma, the discount rate for future rewards")
flags.DEFINE_integer("T", 1e6, "Number of frames to act for")
flags.DEFINE_integer("episodes", 10000, "Number of episodes to act for")
flags.DEFINE_integer("action_override", 0, "Overrides the number of actions provided by the environment")
flags.DEFINE_float("grad_clip", 10, "Clips gradients by their norm")
flags.DEFINE_integer("seed", 0, "Seed for numpy and tf")
flags.DEFINE_integer("checkpoint", 1e5, "How often to save the global model")
flags.DEFINE_integer("xp", 1e5, "Size of the experience replay")
flags.DEFINE_float("epsilon_start", 1, "Value of epsilon to start with")
flags.DEFINE_float("epsilon_finish", 0.01, "Final value of epsilon to anneal to")
flags.DEFINE_integer("target", 100, "After how many steps to update the target network")
flags.DEFINE_boolean("double", True, "Double DQN or not")
flags.DEFINE_integer("batch", 64, "Minibatch size")
flags.DEFINE_integer("summary", 5, "After how many steps to log summary info")

FLAGS = flags.FLAGS
ENV_NAME = FLAGS.env
env = gym.make(ENV_NAME)

if FLAGS.action_override > 0:
    ACTIONS = FLAGS.action_override
else:
    ACTIONS = env.action_space.n
DOUBLE_DQN = FLAGS.double
SEED = FLAGS.seed
LR = FLAGS.learning_rate
NAME = FLAGS.name
EPISODES = FLAGS.episodes
EPSILON_START = FLAGS.epsilon_start
EPSILON_FINISH = FLAGS.epsilon_finish
XP_SIZE = FLAGS.xp
GAMMA = FLAGS.gamma
BATCH_SIZE = FLAGS.batch
TARGET_UPDATE = FLAGS.target
SUMMARY_UPDATE = FLAGS.summary
CLIP_VALUE = FLAGS.grad_clip
CHECKPOINT_INTERVAL = FLAGS.checkpoint
if FLAGS.logdir != "":
    LOGDIR = FLAGS.logdir
else:
    LOGDIR = "Logs/{}_{}".format(NAME, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

if not os.path.exists("{}/ckpts/".format(LOGDIR)):
    os.makedirs("{}/ckpts".format(LOGDIR), exist_ok=True)

print("\n--------Info--------")
print("Logdir:", LOGDIR)
print("--------------------\n")

# TODO: Prioritized experience replay
replay = ExperienceReplay(XP_SIZE)

with tf.Graph().as_default():
    with tf.Session() as sess:

        # Seed numpy and tensorflow
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        test_state = env.reset()

        dqn = model(name="DQN", actions=ACTIONS)
        target_dqn = model(name="Target_DQN", actions=ACTIONS)

        dqn_inputs = dqn["Input"]
        target_dqn_input = dqn["Input"]
        dqn_qvals = dqn["Q_Values"]
        target_dqn_qvals = dqn["Q_Values"]
        dqn_vars = dqn["Variables"]
        target_dqn_vars = target_dqn["Variables"]
        dqn_targets = dqn["Targets"]
        dqn_actions = dqn["Actions"]
        dqn_summary_loss = dqn["Loss_Summary"]
        dqn_summary_qvals = dqn["QVals_Summary"]

        with tf.name_scope("Sync_Target_DQN"):
            sync_vars = []
            for (ref, val) in zip(target_dqn_vars, dqn_vars):
                sync_vars.append(tf.assign(ref, val))

        T = 0
        qval_loss = dqn["Q_Loss"]

        optimiser = tf.train.AdamOptimizer(LR)
        grads_vars = optimiser.compute_gradients(qval_loss)
        clipped = []
        for grad, var in grads_vars:
            if grad is not None:
                clipped.append((tf.clip_by_norm(grad, CLIP_VALUE), var))
            else:
                clipped.append((None, var))
        minimise_op = optimiser.apply_gradients(clipped)

        episode_reward = tf.placeholder(tf.float32)
        reward_summary = tf.scalar_summary("Episode Reward", episode_reward)
        tf_epsilon = tf.placeholder(tf.float32)
        epsilon_summary = tf.scalar_summary("Epsilon", tf_epsilon)

        sess.run(tf.initialize_all_variables())
        sess.run(sync_vars)

        writer = tf.train.SummaryWriter("{}/tb_logs/dqn_agent".format(LOGDIR), graph=sess.graph)
        saver = tf.train.Saver(max_to_keep=None, var_list=dqn_vars)

        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        frames = 0

        for episode in tqdm(range(1, EPISODES + 1)):

            s_t = env.reset()
            episode_finished = False

            epsilon = EPSILON_FINISH + (EPSILON_START - EPSILON_FINISH) * ((EPISODES - episode) / EPISODES)

            ep_reward = 0
            while not episode_finished:
                # env.render()
                # TODO: Exploratory phase
                q_vals, qvals_summary = sess.run([dqn_qvals, dqn_summary_qvals], feed_dict={dqn_inputs: [s_t]})
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_vals[0, :])

                s_new, r_t, episode_finished, _ = env.step(action)
                ep_reward += r_t
                replay.Add_Exp(s_t, action, r_t, s_new, episode_finished)
                s_t = s_new

                batch = replay.Sample(BATCH_SIZE)
                # Create targets from the batch
                old_states = list(map(lambda tups: tups[0], batch))
                new_states = list(map(lambda tups: tups[3], batch))
                current_qvals, target_qvals = sess.run([dqn_qvals, target_dqn_qvals], feed_dict={target_dqn_input: new_states, dqn_inputs: old_states})
                new_state_qvals = sess.run(dqn_qvals, feed_dict={dqn_inputs: new_states})
                q_targets = []
                actions = []
                for batch_item, target_qval, double_qvals in zip(batch, target_qvals, new_state_qvals):
                    st, at, rt, snew, terminal = batch_item
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
                loss_summary, _ = sess.run([dqn_summary_loss, minimise_op], feed_dict={dqn_inputs: old_states, dqn_targets: q_targets, dqn_actions: actions})
                frames += 1
                T += 1

                if frames % TARGET_UPDATE == 0:
                    # print("Before", sess.run(target_dqn_qvals, feed_dict={target_dqn_input: test_state[np.newaxis, :]}))
                    sess.run(sync_vars)
                    # print("After", sess.run(target_dqn_qvals, feed_dict={target_dqn_input: test_state[np.newaxis, :]}))

                if frames % SUMMARY_UPDATE == 0:
                    writer.add_summary(loss_summary, global_step=T)
                    writer.add_summary(qvals_summary, global_step=T)

                if frames % CHECKPOINT_INTERVAL == 0:
                    saver.save(sess=sess, save_path="{}/ckpts/dqn_vars".format(LOGDIR), global_step=T)

            r_summary, e_summary = sess.run([reward_summary, epsilon_summary], feed_dict={tf_epsilon: epsilon, episode_reward: ep_reward})
            writer.add_summary(r_summary, global_step=episode)
            writer.add_summary(e_summary, global_step=episode)

            # TODO: Evaluation episodes with just greedy policy, track qvalues over the episode

        # Print information specific to Lake
        for i in range(16):
            qs = sess.run(dqn_qvals, feed_dict={dqn_inputs: [i]})
            print("{}: {}".format(i, qs))
