import tensorflow as tf
import tflearn


def model(name="Model", actions=1, beta=0.01):
    with tf.name_scope(name):
        # Last 4 observed frames with all 3 colour channels resized to 105x80 from 210x160
        obs = tf.placeholder(tf.float32, shape=[None, 105, 80, 12], name="Observation_Input")
        net = tflearn.conv_2d(obs, 16, 8, 4, activation="relu", name="Conv1")
        net = tflearn.conv_2d(net, 32, 4, 2, activation="relu", name="Conv2")
        net = tflearn.fully_connected(net, 256, activation="relu", weights_init="xavier", name="FC1")
        with tf.name_scope("value"):
            value = tflearn.fully_connected(net, 1, activation="linear", weights_init="xavier", name="Value")
        with tf.name_scope("policy"):
            policy = tflearn.fully_connected(net, actions, activation="softmax", weights_init="xavier", name="Policy")

        # Clip to avoid NaNs
        policy = tf.clip_by_value(policy, 1e-10, 1)

        value_target = tf.placeholder(tf.float32, shape=[None, 1], name="Value_Target")
        value_error = value_target - value
        # Apparently they multiply by 0.5 in the Async paper
        value_loss = 0.5 * tf.reduce_sum(tf.square(value_error))

        log_policy = tf.log(policy)
        action_index = tf.placeholder(tf.float32, shape=[None, actions], name="Action_Taken")
        # We have the Probability distribution for the actions, we want the probability of taking the
        # action that was actually taken
        # tf.mul is elementwise multiplication, hence then reduce_sum.
        # reduction_index = 1 since dim 0 is for batches
        log_probability_of_action = tf.reduce_sum(log_policy * action_index, reduction_indices=1)

        policy_entropy = -tf.reduce_sum(policy * log_policy)

        advantage_estimate = value_error

        policy_loss = -tf.reduce_sum(log_probability_of_action * advantage_estimate + beta * policy_entropy)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        p_vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{}/policy".format(name))
        v_vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{}/value".format(name))
        policy_variables = [var for var in variables if var not in v_vs]
        value_variables = [var for var in variables if var not in p_vs]

    # Create a dictionary to hold all these variables
    dict = {}
    dict["Input"] = obs
    dict["Action_Index"] = action_index
    dict["Value_Target"] = value_target
    dict["Value"] = value
    dict["Policy"] = policy
    dict["Value_Loss"] = value_loss
    dict["Policy_Loss"] = policy_loss
    dict["Model_Variables"] = variables
    dict["Value_Variables"] = value_variables
    dict["Policy_Variables"] = policy_variables
    dict["Policy_Entropy"] = policy_entropy

    return dict
