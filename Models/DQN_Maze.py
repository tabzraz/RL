import tensorflow as tf
import tflearn


def model(name="Model", actions=2):
    with tf.name_scope(name):
        inputs = tf.placeholder(tf.float32, shape=[None, 10, 10, 1], name="Observation_Input")

        # Truncated normal (Defauly init for tflearn) is good enough
        net = tflearn.fully_connected(inputs, 256, activation="relu", name="FC1")
        net = tflearn.fully_connected(net, 128, activation="relu", name="FC2")
        # net = tflearn.fully_connected(net, 128, activation="relu", weights_init=w_init, bias_init=b_init, name="FC1")
        q_values = tflearn.fully_connected(net, actions, activation="linear")

        action_index = tf.placeholder(tf.float32, shape=[None, actions])
        target_q = tf.placeholder(tf.float32, shape=[None, actions])
        q_error = action_index * (q_values - target_q)
        q_loss = tf.reduce_mean(tf.square(q_error))

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        # Summaries
        loss_summary = tf.scalar_summary("Q_Loss", q_loss)
        qval_summaries = []
        for i in range(actions):
            qval_summaries.append(tf.scalar_summary("Action {} QValue".format(i), q_values[0, i]))
        average_qval = tf.reduce_mean(q_values)
        avg_qvals_summary = tf.scalar_summary("Average QValue", average_qval)
        qvals_histogram = tf.histogram_summary("Q Values", q_values)
        qval_summaries.append(qvals_histogram)
        qval_summaries.append(avg_qvals_summary)
        qvals_summary = tf.merge_summary(qval_summaries)

    dict = {}
    dict["Input"] = inputs
    dict["Q_Values"] = q_values
    dict["Q_Loss"] = q_loss
    dict["Variables"] = variables
    dict["Targets"] = target_q
    dict["Actions"] = action_index
    dict["Loss_Summary"] = loss_summary
    dict["QVals_Summary"] = qvals_summary

    return dict
