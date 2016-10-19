import numpy as np
import gym
import tensorflow as tf
import threading
import time
from Models.A3C_Atari import model as model
import Envs

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Initial Learning Rate")
flags.DEFINE_integer("actors", 16, "Number of actor threads to use")
flags.DEFINE_float("gamma", 0.99, "Gamma, the discount rate for future rewards")
flags.DEFINE_integer("t_max", 50e6, "Number of frames to run for")
flags.DEFINE_string("env", "Tabz_SpaceInvaders-v0", "Name of OpenAI gym environment to use")
flags.DEFINE_integer("action_override", 0, "Overrides the number of actions provided by the environment")
flags.DEFINE_float("beta", 0.01, "Used to regularise the policy loss via the entropy")
flags.DEFINE_float("grad_clip", 10, "Clips gradients to norm 10")
flags.DEFINE_string("logdir", "Logs", "Directory to put logs (including tensorboard logs)")
flags.DEFINE_integer("episode_t_max", 500, "Maximum number of frames an actor should act in an episode for")
flags.DEFINE_integer("eval_interval", 1e4, "Rough number of timesteps to wait until evaluating the global model")
flags.DEFINE_integer("eval_runs", 3, "Number of runs to average over for evaluation")
flags.DEFINE_integer("eval_t_max", 1000, "Max frames to run an episode for during evaluation")
FLAGS = flags.FLAGS
# Parameters
# TODO: Use tf.flags to make cmd line configurable
ENV_NAME = FLAGS.env
T = 1
T_MAX = FLAGS.t_max
GAMMA = FLAGS.gamma
if FLAGS.action_override > 0:
    ACTIONS = FLAGS.action_override
else:
    # Make the desired env and check its action space
    test_env = gym.make(ENV_NAME)
    ACTIONS = test_env.action_space.n
    test_env.close()
BETA = FLAGS.beta
LR = FLAGS.learning_rate
THREADS = FLAGS.actors
CLIP_VALUE = FLAGS.grad_clip
LOGDIR = FLAGS.logdir
EPISODE_T_MAX = FLAGS.episode_t_max
EVAL_INTERVAL = FLAGS.eval_interval
EVAL_RUNS = FLAGS.eval_runs
EVAL_T_MAX = FLAGS.eval_t_max


def sample(sess_probs):
    """
    Return an action index by sampling from a policy distribution

    Args:
    sess_probs - Tensorflow vector produced from sess.run(policy,...) of shape [1, P_n]
    """
    # Probably don't need to try and normalise here since we are clipping the output of the softmax
    # p = sess_probs[0, :] / np.sum(sess_probs[0, :])
    p = sess_probs[0, :]

    # Too avoid sum(p) > 1, if sum(p)<1 then the np multinomial will correct it
    p -= np.finfo(np.float32).eps
    one_hot_index = np.random.multinomial(1, p)
    index = np.where(one_hot_index > 0)
    index = index[0][0]
    return index


def actor(env, model, t_max, sess, update_global_model, sync_vars):
    """
    The actor in the Async RL framework
    """
    global T, T_MAX, GAMMA, ACTIONS

    # Get useful values out from the model
    inputs = model["Input"]
    action_index = model["Action_Index"]
    value_target = model["Value_Target"]
    value = model["Value"]
    policy = model["Policy"]

    while T < T_MAX:

        # Copy the parameters of the global model
        sess.run(sync_vars)

        states = []
        actions = []
        rewards = []

        t = 0

        s_t = env.reset()

        episode_finished = False
        while (not episode_finished) and t < t_max:
            # env.render()
            policy_distrib_sess = sess.run(policy, feed_dict={inputs: s_t[np.newaxis, :]})
            a_t_index = sample(policy_distrib_sess)

            s_t, r_t, episode_finished, _ = env.step(a_t_index)

            # One hot encoding for a_t
            a_t = np.zeros(ACTIONS)
            a_t[a_t_index] = 1

            states.append(s_t)
            actions.append(a_t)
            rewards.append(r_t)

            # TODO: Maybe do this with a lock for safe concurrent writes?
            T += 1
            t += 1

        if episode_finished:
            R_t = 0
        else:
            # Bootstrap
            R_t = sess.run(value, feed_dict={inputs: s_t[np.newaxis, :]})[0, 0]

        targets = []
        for i in reversed(range(0, t)):
            R_t = rewards[i] + GAMMA * R_t
            targets.append(np.array([R_t]))
        targets.reverse()

        # Update the central global model via our gradients
        sess.run(update_global_model, feed_dict={inputs: states, action_index: actions, value_target: targets})


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


with tf.Graph().as_default():
    with tf.Session() as sess:

        # Shared optimisers for policy and value
        optimiser = tf.train.AdamOptimizer(LR, use_locking=False)
        global_model = model(name="Global_Model", actions=ACTIONS, beta=BETA)

        global_vars = global_model["Model_Variables"]

        envs = []
        models = []
        update_global_ops = []
        sync_var_ops = []

        for i in range(THREADS):
            envs.append(gym.make(ENV_NAME))
            actor_model = model(name="Actor_{}".format(i + 1), actions=ACTIONS, beta=BETA)
            models.append(actor_model)

            # (obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, variables) = mm
            policy_loss = actor_model["Policy_Loss"]
            value_loss = actor_model["Value_Loss"]
            actor_variables = actor_model["Model_Variables"]

            policy_grads = tf.gradients(policy_loss, actor_variables)
            clipped_policy_grads = []
            for grad in policy_grads:
                if grad is not None:
                    clipped_policy_grads.append(tf.clip_by_norm(grad, CLIP_VALUE))
                else:
                    clipped_policy_grads.append(None)
            policy_grad_vars = zip(clipped_policy_grads, global_vars)
            update_policy_grads = optimiser.apply_gradients(policy_grad_vars)

            value_grads = tf.gradients(value_loss, actor_variables)
            clipped_value_grads = []
            for grad in value_grads:
                if grad is not None:
                    clipped_value_grads.append(tf.clip_by_norm(grad, CLIP_VALUE))
                else:
                    clipped_value_grads.append(None)
            value_grad_vars = zip(clipped_value_grads, global_vars)
            update_value_grads = optimiser.apply_gradients(value_grad_vars)

            update_global_model_op = tf.group(update_policy_grads, update_value_grads)
            update_global_ops.append(update_global_model_op)

            sync_vars = []
            for (ref, val) in zip(actor_variables, global_vars):
                sync_vars.append(tf.assign(ref, val))
            sync_var_ops.append(sync_vars)

        sess.run(tf.initialize_all_variables())

        writer = tf.train.SummaryWriter("{}/tblogs".format(LOGDIR), graph=sess.graph)

        threads = [threading.Thread(target=actor, args=(envs[i], models[i], EPISODE_T_MAX, sess, update_global_ops[i], sync_var_ops[i])) for i in range(THREADS)]

        # TODO: Print some more params here as well
        print("T_MAX: {:,}".format(int(T_MAX)))

        def print_time():
            start_time = time.time()
            while T < T_MAX:
                time_elapsed = time.time() - start_time
                time_left = time_elapsed * (T_MAX - T) / T
                # Just in case, 100 days is the upper limit
                time_left = min(time_left, 60 * 60 * 24 * 100)
                print("\x1b[K", "T:", T, " Elapsed Time:", time_str(time_elapsed), " Left:", time_str(time_left), end="\r")
                time.sleep(1)

        threads.append(threading.Thread(target=print_time))

        for t in threads:
            t.daemon = True
            time.sleep(1)
            t.start()

        reward_tf = tf.placeholder(tf.float32)
        reward_summary = tf.scalar_summary("Average Reward", reward_tf)

        env = gym.make(ENV_NAME)
        eval_model = model(name="Eval_Model", actions=ACTIONS)
        eval_vars = eval_model["Model_Variables"]
        eval_policy = eval_model["Policy"]
        eval_inputs = eval_model["Input"]

        sync_vars = []
        for (ref, val) in zip(eval_vars, global_vars):
            sync_vars.append(tf.assign(ref, val))
        # env.render()
        TLast = -1e7
        while T < T_MAX:
            if T - TLast > EVAL_INTERVAL:
                T_Record = T
                sess.run(sync_vars)
                returns = []
                for _ in range(EVAL_RUNS):
                    R_t = 0
                    s_t = env.reset()

                    episode_finished = False
                    t = 0
                    while (not episode_finished) and (t < EVAL_T_MAX):
                        # env.render()
                        policy_distrib_sess = sess.run(eval_policy, feed_dict={eval_inputs: s_t[np.newaxis, :]})
                        a_t_index = sample(policy_distrib_sess)
                        s_t, r_t, episode_finished, _ = env.step(a_t_index)
                        R_t = r_t + R_t * GAMMA
                        t += 1
                    returns.append(R_t)
                avg_reward = sum(returns) / EVAL_RUNS
                r = sess.run(reward_summary, feed_dict={reward_tf: avg_reward})
                writer.add_summary(r, global_step=T_Record)
                TLast = T
            time.sleep(1)

        for t in threads:
            t.join()
