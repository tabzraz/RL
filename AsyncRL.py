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
    name = "ID_" + str(scope)
    with tf.name_scope(name):
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
        # Maybe change this so that we dont computer the gradient of (value_target - value)
        advantage_no_grad = tf.stop_gradient(value_target - value)
        policy_loss = -log_probability_of_action * advantage_no_grad

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    return obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, variables


def actor(env, id, model, sess, update_policy_gradients, update_value_gradients, t_max, sync_vars):
    global T, T_MAX, GAMMA
    local_model = model
    (obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, _) = local_model

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
        sess.run([update_policy_gradients, update_value_gradients], feed_dict={obs: states, action_index: actions, value_target: targets})

        sess.run(sync_vars)
        # TODO: Rest of stuff here


# Setup the threads
with tf.Graph().as_default():
    with tf.Session() as sess:
        # Shared optimisers for policy and value
        global_policy_optimiser = tf.train.RMSPropOptimizer(0.1)
        global_value_optimiser = tf.train.RMSPropOptimizer(0.1)
        global_model = model()
        (_,_,_,_,_,_,_,_,global_vars) = global_model

        envs = []
        models = []
        policies = []
        values = []
        variables = []

        for i in range(THREADS):
            envs.append(gym.make(ENV_NAME))
            mm = model(i + 1)
            models.append(mm)
            (obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, variables) = mm
            policy_grads = tf.gradients(policy_loss, variables)
            policy_grad_vars = zip(policy_grads, global_vars)
            update_policy_grads = global_policy_optimiser.apply_gradients(policy_grad_vars)
            policies.append(update_policy_grads)
            value_grads = tf.gradients(value_loss, variables)
            value_grad_vars = zip(value_grads, global_vars)
            update_value_grads = global_value_optimiser.apply_gradients(value_grad_vars)
            values.append(update_value_grads)
            sync_vars = []
            for (ref, val) in zip():
                sync_vars.append(tf.assign(ref, val))
            variables.append(sync_vars)

        sess.run(tf.initialize_all_variables())

        tf.train.SummaryWriter("test_graph", graph=sess.graph)

        threads = [threading.Thread(target=actor, args=(envs[i], i, models[i], sess, policies[i], values[i], 500, variables[i]), daemon=True) for i in range(THREADS)]
        # Start the actors
        for t in threads:
            t.start()
        for t in threads:
            t.join()
