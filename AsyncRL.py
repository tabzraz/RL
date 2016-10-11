import numpy as np
import gym
import tensorflow as tf
import tflearn
import threading

# Parameters
# TODO: Use tf.flags to make cmd line configurable
THREADS = 1
ENV_NAME = "CartPole-v0"
T = 0
T_MAX = 1e6
GAMMA = 0.99


def model(scope=0):
    with tf.name_scope("ID_" + str(scope)):
        obs = tf.placeholder(tf.float32, shape=[None, 4], name="Observation_Input")
        net = tflearn.fully_connected(obs, 128, activation="relu", weights_init="variance_scaling", name="FC")
        value = tflearn.fully_connected(net, 1, activation="linear", weights_init="variance_scaling", name="Value")
        policy = tflearn.fully_connected(net, 2, activation="softmax", weights_init="variance_scaling", name="Policy")

        value_target = tf.placeholder(tf.float32, shape=[None, 1], name="Value_Target")
        value_error = value_target - value
        value_loss = tf.reduce_sum(tf.square(value_error))

        log_policy = tf.log(policy)
        action_index = tf.placeholder(tf.float32, shape=[None, 2], name="Action_Taken")
        # We have the Probability distribution for the actions, we want the probability of taking the
        # action that was actually taken
        # tf.mul is elementwise multiplication, hence then reduce_sum.
        # reduction_index = 1 since dim 0 is for batches
        log_probability_of_action = tf.reduce_sum(log_policy * action_index, reduction_indices=1)
        # Don't include the critic here because it shares parameters with policy.
        # Multiply by the values when calculating the gradient
        policy_loss = -log_probability_of_action  # * (value_target - value)

    return obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss


def actor(env, id, model, sess, compute_policy_gradients, compute_value_gradients, t_max, apply_gradients):
    global T, T_MAX, GAMMA
    local_model = model
    (obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss) = local_model

    while T < T_MAX:
        states = []
        actions = []
        rewards = []

        t = 0
        s_t = env.reset()
        episode_finished = False
        while (not episode_finished) and t < t_max:
            # env.render()
            policy_distrib = sess.run(policy, feed_dict={obs: s_t[np.newaxis,:]})
            a_t_index = np.random.choice(2, p=policy_distrib[0,:])
            s_t, r_t, episode_finished, _ = env.step(a_t_index)
            # One hot encoding for a_t
            a_t = np.zeros(2)
            a_t[a_t_index] = 1

            states.append(s_t)
            actions.append(a_t)
            rewards.append(r_t)

            t += 1
            # TODO: Do this with a lock for safe concurrent writes
            T += 1

        if episode_finished:
            R_t = 0
        else:
            # Bootstrap
            R_t = sess.run(value, feed_dict={obs: s_t})

        targets = []
        for i in reversed(range(0, t)):
            R_t = rewards[i] + GAMMA * R_t
            targets.append(np.array([R_t]))
        targets.reverse()

        # print(states, "\n", actions, "\n", targets)
        value_errors = \
            sess.run([value_error],
                     feed_dict={obs: states, action_index: actions, value_target: targets})
        policy_gradients_incomplete = compute_policy_gradients(policy_loss)
        value_gradients = compute_value_gradients(value_loss)

        policy_gradients = policy_gradients_incomplete * value_errors
        apply_gradients(policy_gradients, value_gradients)

        # TODO: Rest of stuff here


# Setup the threads
with tf.Graph().as_default():
    with tf.Session() as sess:
        # Shared optimisers for policy and value
        global_policy_optimiser = tf.train.RMSPropOptimizer(0.1)
        global_value_optimiser = tf.train.RMSPropOptimizer(0.1)
        global_model = model()

        envs = []
        models = []
        policies = []
        values = []

        for i in range(THREADS):
            envs.append(gym.make(ENV_NAME))
            mm = model(i + 1)
            models.append(mm)
            (obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss) = mm
            policies.append(global_policy_optimiser.compute_gradients)
            values.append(global_value_optimiser.compute_gradients)

        print(values)

        def apply_gradients(policy_grads, value_grads):
            global_policy_optimiser.apply_gradients(policy_grads)
            global_value_optimiser.apply_gradients(value_grads)

        sess.run(tf.initialize_all_variables())

        tf.train.SummaryWriter("test_graph", graph=sess.graph)

        threads = [threading.Thread(target=actor, args=(envs[i], i, models[i], sess, policies[i], values[i], 500, apply_gradients), daemon=True) for i in range(THREADS)]
        # Start the actors
        for t in threads:
            t.start()
        for t in threads:
            t.join()
