import tensorflow as tf
import tflearn


def model(name="Model", actions=2, beta=0.01):
    with tf.name_scope(name):
        # Last 4 observed frames with all 3 colour channels resized to 105x80 from 210x160
        obs = tf.placeholder(tf.float32, shape=[None, 4], name="Observation_Input")
        net = tflearn.fully_connected(obs, 16, activation="relu", weights_init="xavier", name="FC1")
        value = tflearn.fully_connected(net, 1, activation="linear", weights_init="xavier", name="Value")
        policy = tflearn.fully_connected(net, actions, activation="softmax", weights_init="xavier", name="Policy")

        # Clip to avoid NaNs
        policy = tf.clip_by_value(policy, 1e-10, 1)

        value_target = tf.placeholder(tf.float32, shape=[None, 1], name="Value_Target")
        value_error = value_target - value
        # Apparently they multiply by 0.5 in the Async paper
        value_loss = 0.5 * tf.reduce_sum(tf.square(value_error))

        log_policy = tf.log(policy)
        action_index = tf.placeholder(tf.float32, shape=[None, actions], name="Action_Taken")

        log_probability_of_action = tf.reduce_sum(log_policy * action_index, reduction_indices=1)

        policy_entropy = -tf.reduce_sum(policy * log_policy)

        advantage_no_grad = tf.stop_gradient(value_target - value)

        policy_loss = -(log_probability_of_action * advantage_no_grad + beta * policy_entropy)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

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

    return dict
