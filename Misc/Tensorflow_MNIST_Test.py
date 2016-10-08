import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data
# import os

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Graph().as_default():

    inputs = tf.placeholder(tf.float32, shape=[None, 28 * 28], name="Flattened_input")
    # dropout_keep_prob = tf.placeholder(tf.float32, name="Dropout_Probability")
    labels = tf.placeholder(tf.float32, shape=[None, 10], name="Labels")

    inputs_as_image = tf.reshape(inputs, [-1, 28, 28, 1])
    net = tflearn.conv_2d(inputs_as_image, 32, 3, activation="relu", weights_init="variance_scaling", name="Conv1")
    net = tflearn.conv_2d(net, 32, 3, activation="relu", weights_init="variance_scaling", name="Conv2")
    net = tflearn.fully_connected(net, 256, activation="relu", weights_init="variance_scaling", name="FC")
    linear_output = tflearn.fully_connected(net, 10, weights_init="variance_scaling", name="Logits")
    softmax_output = tflearn.activations.softmax(linear_output)
    prediction = tf.argmax(softmax_output, 1, name="Predictions")

    with tf.name_scope("Loss"):
        cross_entropy_losses = tf.nn.softmax_cross_entropy_with_logits(linear_output, labels)
        loss = tf.reduce_mean(cross_entropy_losses)
        loss_summary = tf.scalar_summary("Loss", loss)

    with tf.name_scope("Accuracy"):
        correct_preds = tf.equal(prediction, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        accuracy_summary = tf.scalar_summary("Accuracy", accuracy)

    summaries_op = tf.merge_summary([loss_summary, accuracy_summary])
    summary_dir = "/home/tabz/tmp/summaries/name3"

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimiser = tf.train.AdamOptimizer()
    grads_and_vars = optimiser.compute_gradients(loss)
    apply_grads_op = optimiser.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        summary_writer_train = tf.train.SummaryWriter(summary_dir + "_train", graph=sess.graph)
        summary_writer_val = tf.train.SummaryWriter(summary_dir + "_validation", graph=sess.graph)

        for i in range(1, 1001):
            x, y = mnist.train.next_batch(256)
            feed_dict_train = {inputs: x, labels: y}
            _, step, l, a, summaries = sess.run([apply_grads_op, global_step, loss, accuracy, summaries_op], feed_dict_train)

            summary_writer_train.add_summary(summaries, step)

            if i % 100 == 0:
                feed_dict_val = {inputs: mnist.validation.images, labels: mnist.validation.labels}
                val_acc, val_sum = sess.run([accuracy, summaries_op], feed_dict_val)
                summary_writer_val.add_summary(val_sum, step)
                print("Step: {:>4}, Loss: {:.10f}, Acc: {:>7.2%}, Val Acc: {:>7.2%}".format(step, l, a, val_acc))
