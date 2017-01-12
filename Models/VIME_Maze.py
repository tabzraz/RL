import tensorflow as tf
from Bayesian.Bayesian_Conv_Layer import Bayesian_Conv
from Bayesian.Bayesian_DeConv_Layer import Bayesian_DeConv
from Bayesian.Bayesian_FC_Layer import Bayesian_FC
from Bayesian.Bayesian_Net import Bayesian_Net, log_gaussian_pdf
from math import ceil


def model(name="Exploration_Model", size=1, actions=4):
    with tf.name_scope(name):
        inputs = tf.placeholder(tf.float32, shape=[None, size * 7, size * 7, 1], name="State")
        action = tf.placeholder(tf.float32, shape=[None, actions], name="Action")
        target = tf.placeholder(tf.float32, shape=[None, size * 7, size * 7, 1], name="Next_State")
        kl_scaling = tf.placeholder(tf.float32, shape=[])
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

        def bnet_prob(pred, target):
            return log_gaussian_pdf(pred, target, 0.1)

        bayesian_net = Bayesian_Net([l1, l2, l3, l4, l5], bnet_prob)

        bnet_loss, kl_loss, data_loss = bayesian_net.loss(sample, target, kl_scaling=kl_scaling, N=8)
        bnet_output = sample(local_reparam_trick=False)

    dict = {}
    dict["Input"] = inputs
    dict["Output"] = bnet_output
    dict["Target"] = target
    dict["Action"] = action
    dict["Loss"] = bnet_loss
    dict["KL_Loss"] = kl_loss
    dict["Data_Loss"] = data_loss

    return dict


def tf_conv_size(W, f, s):
    return ceil((W - f + 1) / s)
