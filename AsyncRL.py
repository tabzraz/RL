import numpy as np
import gym
# import gym_ple
import tensorflow as tf
import tflearn
import threading
import time
from skimage.transform import resize
import datetime

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Initial Learning Rate")
tf.app.flags.DEFINE_integer("actors", 16, "Number of actor threads to use")
FLAGS = tf.app.flags.FLAGS
# Parameters
# TODO: Use tf.flags to make cmd line configurable
ENV_NAME = "Breakout-v0"
T = 1
T_MAX = 5e7
GAMMA = 0.99
ACTIONS = 3
BETA = 0.01
LR = FLAGS.learning_rate
THREADS = FLAGS.actors

print("LR: ", LR)


def model(scope=0):
    global ACTIONS, BETA
    name = "ID_" + str(scope)
    with tf.name_scope(name):
        # Last 4 observed frames with all 3 colour channels
        obs = tf.placeholder(tf.float32, shape=[None, 105, 80, 12], name="Observation_Input")
        # net = tflearn.fully_connected(obs, 64, activation="relu", weights_init="xavier", name="FC")
        net = tflearn.conv_2d(obs, 16, 8, 4, activation="relu")
        net = tflearn.conv_2d(net, 32, 4, 2, activation="relu")
        net = tflearn.fully_connected(net, 256, activation="relu", weights_init="xavier")
        value = tflearn.fully_connected(net, 1, activation="linear", weights_init="xavier", name="Value")
        policy = tflearn.fully_connected(net, ACTIONS, activation="softmax", weights_init="xavier", name="Policy")

        # Clip to avoid NaNs
        policy = tf.clip_by_value(policy, 1e-10, 1)

        value_target = tf.placeholder(tf.float32, shape=[None, 1], name="Value_Target")
        value_error = value_target - value
        # Apparently they multiply by 0.5
        value_loss = 0.5 * tf.reduce_sum(tf.square(value_error))

        log_policy = tf.log(policy)
        action_index = tf.placeholder(tf.float32, shape=[None, ACTIONS], name="Action_Taken")
        # We have the Probability distribution for the actions, we want the probability of taking the
        # action that was actually taken
        # tf.mul is elementwise multiplication, hence then reduce_sum.
        # reduction_index = 1 since dim 0 is for batches
        log_probability_of_action = tf.reduce_sum(log_policy * action_index, reduction_indices=1)

        policy_entropy = -tf.reduce_sum(policy * log_policy)
        # Maybe change this so that we dont computer the gradient of (value_target - value)
        advantage_no_grad = tf.stop_gradient(value_target - value)
        policy_loss = -log_probability_of_action * advantage_no_grad + BETA * policy_entropy

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    return obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, variables


def add_frame(frames, new_frame):
    resized_frame = resize(new_frame, (105, 80))
    new_frames = np.concatenate([frames[:, :, 3:], resized_frame], axis=2)
    return new_frames


def start_frames(frame):
    resized_frame = resize(frame, (105, 80))
    new_frames = np.concatenate([resized_frame for _ in range(4)], axis=2)
    return new_frames


def sample(sess_probs):
    # Probably don't need to try and normalise here since we are clipping the output of the softmax
    # p = sess_probs[0, :] / np.sum(sess_probs[0, :])
    p = sess_probs[0, :]

    # Too avoid sum(p) > 1, if sum(p)<1 then the np multinomial will correct it
    p -= np.finfo(np.float32).eps
    one_hot_index = np.random.multinomial(1, p)
    index = np.where(one_hot_index > 0)
    if index[0].size == 0:
        print(sess_probs, p, one_hot_index)
    index = index[0][0]
    return index


def actor(env, id, model, sess, update_policy_gradients, update_value_gradients, t_max, sync_vars):
    global T, T_MAX, GAMMA, ACTIONS
    local_model = model
    (obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, variables) = local_model

    # Try and decorrelate the updates
    time.sleep(id)

    while T < T_MAX:
        states = []
        actions = []
        rewards = []

        t = 0

        s_t = env.reset()
        frames = start_frames(s_t)

        episode_finished = False
        while (not episode_finished) and t < t_max:
            # env.render()
            policy_distrib_sess = sess.run(policy, feed_dict={obs: frames[np.newaxis, :]})
            a_t_index = sample(policy_distrib_sess)

            s_t, r_t, episode_finished, _ = env.step(a_t_index)
            frames = add_frame(frames, s_t)

            # One hot encoding for a_t
            a_t = np.zeros(ACTIONS)
            a_t[a_t_index] = 1

            states.append(frames)
            actions.append(a_t)
            rewards.append(r_t)

            t += 1
            # TODO: Do this with a lock for safe concurrent writes
            T += 1

        if episode_finished:
            R_t = 0
        else:
            # Bootstrap
            R_t = sess.run(value, feed_dict={obs: frames[np.newaxis, :]})[0, 0]

        targets = []
        for i in reversed(range(0, t)):
            R_t = rewards[i] + GAMMA * R_t
            targets.append(np.array([R_t]))
        targets.reverse()

        # print(states, "\n", actions, "\n", targets)
        sess.run([update_policy_gradients, update_value_gradients], feed_dict={obs: states, action_index: actions, value_target: targets})

        sess.run(sync_vars)

        # print("Id:", id, "-Score:", np.sum(rewards))


def time_str(s):
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


# Setup the threads
with tf.Graph().as_default():
    with tf.Session() as sess:
        # Shared optimisers for policy and value
        optimiser = tf.train.AdamOptimizer(LR, use_locking=False)
        global_model = model()
        (global_obs, _, _, _, global_policy, _, _, _, global_vars) = global_model

        envs = []
        models = []
        policies = []
        values = []
        variables_list = []

        for i in range(THREADS):
            envs.append(gym.make(ENV_NAME))
            mm = model(i + 1)
            models.append(mm)
            (obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, variables) = mm
            policy_grads = tf.gradients(policy_loss, variables)
            clipped_policy_grads = []
            for grad in policy_grads:
                if grad is not None:
                    clipped_policy_grads.append(tf.clip_by_norm(grad, 10))
                else:
                    clipped_policy_grads.append(None)
            policy_grad_vars = zip(clipped_policy_grads, global_vars)
            update_policy_grads = optimiser.apply_gradients(policy_grad_vars)
            policies.append(update_policy_grads)
            value_grads = tf.gradients(value_loss, variables)
            clipped_value_grads = []
            for grad in value_grads:
                if grad is not None:
                    clipped_value_grads.append(tf.clip_by_norm(grad, 10))
                else:
                    clipped_value_grads.append(None)
            value_grad_vars = zip(clipped_value_grads, global_vars)
            update_value_grads = optimiser.apply_gradients(value_grad_vars)
            values.append(update_value_grads)
            sync_vars = []
            for (ref, val) in zip(variables, global_vars):
                sync_vars.append(tf.assign(ref, val))
            variables_list.append(sync_vars)

        sess.run(tf.initialize_all_variables())

        writer = tf.train.SummaryWriter("test_graph", graph=sess.graph)

        # print(len(envs), len(models), len(policies), len(values), len(variables_list))

        threads = [threading.Thread(target=actor, args=(envs[i], i, models[i], sess, policies[i], values[i], 300, variables_list[i])) for i in range(THREADS)]

        print("T_MAX: {:,}".format(int(T_MAX)))

        reward_tf = tf.placeholder(tf.float32)
        reward_summary = tf.scalar_summary("AVg Reward", reward_tf)

        # TODO: Use a checkpointed model or copy the weights when you start evaluating because they will change
        def eval_policy():
            env = gym.make(ENV_NAME)
            # env.render()
            TLast = -1e7
            times = 3
            while T < T_MAX:
                if T - TLast > 1e4:
                    returns = []
                    for _ in range(times):
                        R_t = 0
                        s_t = env.reset()
                        frames = start_frames(s_t)

                        episode_finished = False
                        t = 0
                        while (not episode_finished) and (t < 500):
                            # env.render()
                            policy_distrib_sess = sess.run(policy, feed_dict={obs: frames[np.newaxis, :]})
                            a_t_index = sample(policy_distrib_sess)

                            s_t, r_t, episode_finished, _ = env.step(a_t_index)
                            frames = add_frame(frames, s_t)
                            R_t = r_t + R_t * GAMMA
                            t += 1
                        returns.append(R_t)
                    avg_reward = sum(returns) / times
                    rr = sess.run(reward_summary, feed_dict={reward_tf: avg_reward})
                    writer.add_summary(rr)
                    # print(avg_reward)
                    TLast = T
                time.sleep(1)

        threads.append(threading.Thread(target=eval_policy))

        for t in threads:
            t.daemon = True
            t.start()

        start_time = time.time()

        while T < T_MAX:
            time_elapsed = time.time() - start_time
            time_left = time_elapsed * (T_MAX - T) / T
            # Just in case, 100 days is the upper limit
            time_left = min(time_left, 60 * 60 * 24 * 100)
            print("\x1b[K", "T:", T, " Elapsed Time:", time_str(time_elapsed), " Left:", time_str(time_left), end="\r")
            time.sleep(1)
