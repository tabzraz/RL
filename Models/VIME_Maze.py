import tensorflow as tf
from Bayesian.Bayesian_Conv_Layer import Bayesian_Conv
from Bayesian.Bayesian_DeConv_Layer import Bayesian_DeConv
from Bayesian.Bayesian_FC_Layer import Bayesian_FC
from Bayesian.Bayesian_Net import Bayesian_Net
from math import ceil


def model(name="Exploration_Model", size=1, actions=4):
    with tf.name_scope(name):
        inputs = tf.placeholder(tf.float32, shape=[None, size * 7, size * 7, 1], name="Observation_Input")
        action = tf.placeholder(tf.float32, shape=[None, actions])
        img_size = 7 * size

        l1 = Bayesian_Conv(1, 8, filter_height=3, filter_width=3, filter_stride=2)
        img_size = tf_conv_size(img_size, 3, 2)
        img_size_after_l1 = img_size

        l2 = Bayesian_Conv(8, 8, filter_height=3, filter_width=3, filter_stride=2)
        img_size = tf_conv_size(img_size, 3, 2)

        flattened_image_size = img_size * img_size * 8
        l3 = Bayesian_FC(flattened_image_size + actions, flattened_image_size)

        l4 = Bayesian_DeConv((img_size, img_size), 8, 8, filter_height=3, filter_width=3, filter_stride=2)

        l5 = Bayesian_DeConv((img_size_after_l1, img_size_after_l1), 8, 1, filter_height=3, filter_width=3, filter_stride=2, activation=tf.nn.sigmoid)

        def sample(local_reparam_trick=False):
            net = l1.sample(inputs, local_reparam_trick)
            net = l2.sample(net, local_reparam_trick)
            # Flatten image to put through fc layer
            net = tf.reshape(net, shape=[-1, flattened_image_size])
            net = tf.concat(1, [net, action])
            net = l3.sample(net, local_reparam_trick)
            # Unflatten to deconv
            net = tf.reshape(net, shape=[-1, img_size, img_size, 8])
            net = l4.sample(net, local_reparam_trick)
            net = l5.sample(net, local_reparam_trick)
            return net


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


def tf_conv_size(W, f, s):
    return ceil((W - f + 1) / s)
