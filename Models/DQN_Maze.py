import tensorflow as tf
import tflearn
from Misc.Misc import tf_conv_size
from math import ceil

def model(name="Model", size=1, actions=4):
    with tf.name_scope(name):
        inputs = tf.placeholder(tf.float32, shape=[None, size * 7, size * 7, 1], name="Observation_Input")
        img_size = size * 7

        net = tflearn.conv_2d(inputs, 8, 3, 2, activation="relu", padding="valid", name="Conv1")
        img_size = tf_conv_size(img_size, 3, 2)

        net = tflearn.conv_2d(net, 8, 3, 2, activation="relu", padding="valid", name="Conv2")
        img_size = tf_conv_size(img_size, 3, 2)

        net = tflearn.fully_connected(net, img_size*4, activation="relu")

        q_values = tflearn.fully_connected(net, actions, activation="linear")

        # Dont need dueling yet 
        # v_stream = tflearn.fully_connected(net, ceil(img_size/2), activation="relu")
        # v_stream = tflearn.fully_connected(v_stream, 1, activation="linear")

        # a_stream = tflearn.fully_connected(net, ceil(img_size/2), activation="relu")
        # a_stream = tflearn.fully_connected(a_stream, actions, activation="linear")

        # q_values = v_stream + (a_stream - tf.reduce_mean(a_stream, reduction_indices=1, keep_dims=True))

        action_index = tf.placeholder(tf.float32, shape=[None, actions])
        target_q = tf.placeholder(tf.float32, shape=[None, actions])
        q_error = action_index * (q_values - target_q)
        q_loss = tf.reduce_mean(tf.square(q_error))

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        # Summaries
        loss_summary = tf.summary.scalar("Q_Loss", q_loss)
        qval_summaries = []
        for i in range(actions):
            qval_summaries.append(tf.summary.scalar("Action {} QValue".format(i), q_values[0, i]))
        average_qval = tf.reduce_mean(q_values)
        avg_qvals_summary = tf.summary.scalar("Average QValue", average_qval)
        qvals_histogram = tf.summary.histogram("Q Values", q_values)
        qval_summaries.append(qvals_histogram)
        qval_summaries.append(avg_qvals_summary)
        qvals_summary = tf.summary.merge(qval_summaries)

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
