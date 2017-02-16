import tensorflow as tf
import tflearn
from Misc.Misc import tf_conv_size


def model(name="Model", size=1, actions=4):
    with tf.name_scope(name):
        print("Model: {}".format(name))
        inputs = tf.placeholder(tf.float32, shape=[None, size * 7, size * 7, 1], name="Observation_Input")
        img_size = size * 7
        print("Input: {0}x{0}x{1}".format(size * 7, 1))

        net = tflearn.conv_2d(inputs, nb_filter=4, filter_size=3, strides=2, activation="relu", padding="same", name="Conv_1")
        img_size = tf_conv_size(img_size, 3, 2)
        print("Conv: {0}x{0}x{1}".format(img_size, 4))

        net = tflearn.conv_2d(inputs, nb_filter=8, filter_size=3, strides=2, activation="relu", padding="same", name="Conv_2")
        img_size = tf_conv_size(img_size, 3, 2)
        print("Conv: {0}x{0}x{1}".format(img_size, 8))

        # net = tflearn.conv_2d(inputs, nb_filter=4, filter_size=3, strides=1, activation="relu", padding="same", name="Conv_3")
        # img_size = tf_conv_size(img_size, 3, 1)
        # print("Conv: {0}x{0}x{1}".format(img_size, 4))

        net = tflearn.fully_connected(net, img_size * img_size * 2, activation="relu")
        print("FC: {0} -> {1}".format(img_size * img_size * 8, img_size * img_size * 2))

        q_values = tflearn.fully_connected(net, actions, activation="linear")
        print("FC: {0} -> {1}".format(img_size * img_size * 2, actions))
        print()
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
