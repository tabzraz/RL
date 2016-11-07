import tensorflow as tf
import tflearn


def model(name="Model", actions=4):
    with tf.name_scope(name):
        inputs = tf.placeholder(tf.float32, shape=[None, 105, 80, 4], name="Obs_Input")
        net = tflearn.conv_2d(inputs, 16, 8, 4, activation="relu", name="Conv1")
        net = tflearn.conv_2d(net, 32, 4, 2, activation="relu", name="Conv2")
        net = tflearn.fully_connected(net, 256, activation="relu", name="FC1")

        v_stream = tflearn.fully_connected(net, 64, activation="relu")
        v_stream = tflearn.fully_connected(v_stream, 1, activation="linear")

        a_stream = tflearn.fully_connected(net, 64, activation="relu")
        a_stream = tflearn.fully_connected(a_stream, actions, activation="linear")

        q_values = v_stream + (a_stream - tf.reduce_mean(a_stream, reduction_indices=1, keep_dims=True))

        action_index = tf.placeholder(tf.float32, shape=[None, actions])
        target_q = tf.placeholder(tf.float32, shape=[None, actions])
        q_error = action_index * (q_values - target_q)
        q_loss = tf.reduce_mean(tf.square(q_error))

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        # Summaries
        loss_summary = tf.scalar_summary("Q_Loss", q_loss)
        qval_summaries = []
        # for i in range(actions):
        #    qval_summaries.append(tf.scalar_summary("Action {} QValue".format(i), q_values[0, i]))
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
