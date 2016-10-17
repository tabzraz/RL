import numpy as np
import gym
# import gym_ple
import tensorflow as tf
import tflearn
import threading
import time
# from skimage.transform import resize

# Parameters
# TODO: Use tf.flags to make cmd line configurable
THREADS = 16
ENV_NAME = "FrozenLake-v0"
T = 0
T_MAX = 1e6
GAMMA = 0.99
ACTIONS = 4
BETA = 0.01
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.01, "Initial Learning Rate")
LR = FLAGS.learning_rate

print("LR: ", LR)


def model(scope=0):
    global ACTIONS
    name = "ID_" + str(scope)
    with tf.device('/cpu:0'):
        with tf.name_scope(name):
            obs = tf.placeholder(tf.float32, shape=[None, 16], name="Observation_Input")
            net = tflearn.fully_connected(obs, 64, activation="relu", weights_init="xavier", name="FC")
            value = tflearn.fully_connected(net, 1, activation="linear", weights_init="xavier", name="Value")
            policy = tflearn.fully_connected(net, ACTIONS, activation="softmax", weights_init="xavier", name="Policy")

            tflearn.get_all_variables

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
            # Maybe change this so that we dont computer the gradient of (value_target - value)
            advantage_no_grad = tf.stop_gradient(value_target - value)
            policy_loss = -log_probability_of_action * advantage_no_grad

            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    return obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, variables


def actor(env, id, model, sess, update_policy_gradients, update_value_gradients, t_max, sync_vars):
    global T, T_MAX, GAMMA, ACTIONS
    local_model = model
    (obs, action_index, value_target, value, policy, value_error, value_loss, policy_loss, variables) = local_model

    time.sleep(id)

    while T < T_MAX:
        states = []
        actions = []
        rewards = []

        t = 0
        s_t = env.reset()
        s_t_oh = np.zeros(16)
        s_t_oh[s_t] = 1
        s_t = s_t_oh
        episode_finished = False
        while (not episode_finished) and t < t_max:
            # env.render()
            policy_distrib = sess.run(policy, feed_dict={obs: s_t[np.newaxis,:]})
            a_t_index = np.random.choice(ACTIONS, p=policy_distrib[0,:])
            s_t, r_t, episode_finished, _ = env.step(a_t_index)
            s_t_oh = np.zeros(16)
            s_t_oh[s_t] = 1
            s_t = s_t_oh
            # One hot encoding for a_t
            a_t = np.zeros(ACTIONS)
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
            R_t = sess.run(value, feed_dict={obs: s_t[np.newaxis,:]})[0, 0]
            print(R_t)

        targets = []
        for i in reversed(range(0, t)):
            R_t = rewards[i] + GAMMA * R_t
            targets.append(np.array([R_t]))
        targets.reverse()

        # print(states, "\n", actions, "\n", targets)
        sess.run([update_policy_gradients, update_value_gradients], feed_dict={obs: states, action_index: actions, value_target: targets})

        sess.run(sync_vars)

        # print("Id:", id, "-Score:", np.sum(rewards))
        # TODO: Rest of stuff here


# Setup the threads
with tf.Graph().as_default():
    with tf.Session() as sess:
        # Shared optimisers for policy and value
        learning_rate = tf.train.exponential_decay(1e-1, T, T_MAX, 0.99, staircase=True)
        global_policy_optimiser = tf.train.RMSPropOptimizer(learning_rate)
        global_value_optimiser = tf.train.RMSPropOptimizer(learning_rate)
        global_model = model()
        (global_obs, _, _, _, global_policy, _, _, _, global_vars) = global_model

        envs = []
        models = []
        policies = []
        values = []
        variables_list = []

        for i in range(THREADS):
            with tf.device('/cpu:0'):
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
                update_policy_grads = global_policy_optimiser.apply_gradients(policy_grad_vars)
                policies.append(update_policy_grads)
                value_grads = tf.gradients(value_loss, variables)
                clipped_value_grads = []
                for grad in value_grads:
                    if grad is not None:
                        clipped_value_grads.append(tf.clip_by_norm(grad, 10))
                    else:
                        clipped_value_grads.append(None)
                value_grad_vars = zip(clipped_value_grads, global_vars)
                update_value_grads = global_value_optimiser.apply_gradients(value_grad_vars)
                values.append(update_value_grads)
                sync_vars = []
                for (ref, val) in zip(variables, global_vars):
                    sync_vars.append(tf.assign(ref, val))
                variables_list.append(sync_vars)

        sess.run(tf.initialize_all_variables())

        tf.train.SummaryWriter("test_graph", graph=sess.graph)

        # print(len(envs), len(models), len(policies), len(values), len(variables_list))

        threads = [threading.Thread(target=actor, args=(envs[i], i, models[i], sess, policies[i], values[i], 100, variables_list[i])) for i in range(THREADS)]


        def eval_policy():
            global T, T_MAX, GAMMA, ACTIONS
            # Evaluate policies
            TLast = 0
            env = gym.make(ENV_NAME)
            times = 10
            while T < T_MAX:
                if T- TLast > 1e4:
                    returns = []
                    for _ in range(times):
                        R_t = 0
                        s_t = env.reset()
                        s_t_oh = np.zeros(16)
                        s_t_oh[s_t] = 1
                        s_t = s_t_oh
                        episode_finished = False
                        while (not episode_finished):
                            # env.render()
                            policy_distrib = sess.run(global_policy, feed_dict={global_obs: s_t[np.newaxis,:]})
                            a_t_index = np.random.choice(ACTIONS, p=policy_distrib[0,:])
                            s_t, r_t, episode_finished, _ = env.step(a_t_index)
                            s_t_oh = np.zeros(16)
                            s_t_oh[s_t] = 1
                            s_t = s_t_oh
                            R_t = r_t + R_t * GAMMA
                        returns.append(R_t)
                    print(sum(returns) / times)
                    TLast = T

        threads.append(threading.Thread(target=eval_policy))

        for t in threads:
            t.daemon = True
            t.start()

        while T < T_MAX:
            time.sleep(1)
