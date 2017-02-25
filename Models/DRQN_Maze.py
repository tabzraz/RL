import tensorflow as tf
from Misc.Misc import tf_conv_size


class DRQN:

    def __init__(self, name="Model", size=1, actions=4, lstm_size=16):
        with tf.variable_scope(name):
            print("Model: {}".format(name))
            # Batch x timesteps x w x h x c
            self.batch_size = tf.placeholder(tf.int32, shape=(), name="Batch_Size")
            self.lstm_size = lstm_size
            self.unroll = tf.placeholder(tf.int32, shape=(), name="Unroll")
            self.inputs = tf.placeholder(tf.float32, shape=[None, size * 7, size * 7, 1], name="Observation_Input")
            img_size = size * 7
            # print("Input: {0}x{0}x{1}".format(size * 7, 1))

            # cnn_inputs = tf.reshape(self.inputs, shape=[self.inputs.get_shape()[0] * self.unroll, size * 7, size * 7, 1])
            cnn_inputs = self.inputs
            # net = tflearn.conv_2d(inputs, nb_filter=4, filter_size=3, strides=2, activation="relu", padding="same", name="Conv_1")
            net = tf.layers.conv2d(inputs=cnn_inputs, filters=4, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=tf.nn.relu, name="Conv_1")
            img_size = tf_conv_size(img_size, 3, 2)
            # print("Conv: {0}x{0}x{1}".format(img_size, 4))

            # net = tflearn.conv_2d(net, nb_filter=8, filter_size=3, strides=2, activation="relu", padding="same", name="Conv_2")
            net = tf.layers.conv2d(inputs=net, filters=8, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=tf.nn.relu, name="Conv_2")
            img_size = tf_conv_size(img_size, 3, 2)
            # print("Conv: {0}x{0}x{1}".format(img_size, 8))

            # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size)
            # self.initial_lstm_state_1 = tf.placeholder(tf.float32, shape=[None, lstm_size])
            # self.initial_lstm_state_2 = tf.placeholder(tf.float32, shape=[None, lstm_size])
            # self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state_1, self.initial_lstm_state_2)
            self.initial_lstm_state = self.lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

            # print(self.inputs.get_shape()[0])
            # flattened_net = tf.reshape(net, shape=[-1, img_size * img_size * 8])
            # print(self.unroll)
            lstm_inputs = tf.reshape(net, shape=tf.stack([-1, self.unroll, img_size * img_size * 8]))

            q_values, self.final_states = tf.nn.dynamic_rnn(cell=self.lstm_cell, inputs=lstm_inputs, initial_state=self.initial_lstm_state)

            # Need to apply a final projection onto the q values

            q_values_flattened = tf.reshape(q_values, shape=[-1, lstm_size])
            self.q_values = tf.layers.dense(inputs=q_values_flattened, units=actions, activation=tf.identity, name="Q_Values")
            # # Unroll the lstm
            # q_values_list = []
            # lstm_state = self.initial_lstm_state
            # for i in range(self.unroll):
            #     output, lstm_state = lstm(lstm_inputs[:, i], lstm_state)
            #     q_value = tf.layers.dense(inputs=output, units=actions, activation="linear", name="Q_Values")
            #     q_values_list.append[q_value]
            # self.q_values = tf.stack(q_values_list, axis=1)

            # self.final_lstm_state = lstm_state

            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
            self.action_index = tf.placeholder(tf.float32, shape=[None, actions])
            self.target_q = tf.placeholder(tf.float32, shape=[None, actions])
            q_error = self.action_index * (self.q_values - self.target_q)
            self.q_loss = tf.reduce_mean(tf.square(q_error))

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

            # Summaries
            self.loss_summary = tf.summary.scalar("Q_Loss", self.q_loss)
            qval_summaries = []
            for i in range(actions):
                qval_summaries.append(tf.summary.scalar("Action {} QValue".format(i), self.q_values[0, i]))
            average_qval = tf.reduce_mean(self.q_values)
            avg_qvals_summary = tf.summary.scalar("Average QValue", average_qval)
            qvals_histogram = tf.summary.histogram("Q Values", self.q_values)
            qval_summaries.append(qvals_histogram)
            qval_summaries.append(avg_qvals_summary)
            self.qvals_summary = tf.summary.merge(qval_summaries)

    def zero_state(self, batch_size):
        return self.lstm_cell.zero_state(batch_size, dtype=tf.float32)


    # def model(name="Model", size=1, actions=4):
    #     with tf.name_scope(name):
    #         print("Model: {}".format(name))
    #         # Batch x timesteps x w x h x c
    #         unroll = tf.placeholder(tf.int16, shape=[])
    #         inputs = tf.placeholder(tf.float32, shape=[None, unroll, size * 7, size * 7, 1], name="Observation_Input")
    #         img_size = size * 7
    #         # print("Input: {0}x{0}x{1}".format(size * 7, 1))

    #         cnn_inputs = tf.reshape(inputs, shape=[inputs.get_shape()[0] * unroll, size * 7, size * 7, 1])
    #         # net = tflearn.conv_2d(inputs, nb_filter=4, filter_size=3, strides=2, activation="relu", padding="same", name="Conv_1")
    #         net = tf.layers.conv_2d(inputs=cnn_inputs, filters=4, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=tf.nn.relu, name="Conv_1")
    #         img_size = tf_conv_size(img_size, 3, 2)
    #         # print("Conv: {0}x{0}x{1}".format(img_size, 4))

    #         # net = tflearn.conv_2d(net, nb_filter=8, filter_size=3, strides=2, activation="relu", padding="same", name="Conv_2")
    #         net = tf.layers.conv_2d(inputs=net, filters=8, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=tf.nn.relu, name="Conv_2")
    #         img_size = tf_conv_size(img_size, 3, 2)
    #         # print("Conv: {0}x{0}x{1}".format(img_size, 8))

    #         lstm = tf.contrib.rnn.BasicLSTMCell(16)
    #         lstm_state = tf.zeros([None, lstm.state_size])

    #         lstm_inputs = tf.reshape(net, shape=[inputs.get_shape()[0], unroll, None])

    #         # Unroll the lstm
    #         q_values_list = []
    #         for i in range(unroll):
    #             output, lstm_state = lstm(lstm_inputs[:, i], lstm_state)
    #             q_value = tf.layers.dense(inputs=output, units=actions, activation="linear", name="Q_Values")
    #             q_values_list.append[q_value]
    #         q_values = tf.stack(q_values_list, axis=1)

    #         final_lstm_state = lstm_state

    #         # net = tf.layers.dense(inputs=net, units=img_size * img_size * 2, activation="relu", name="FC_1")
    #         # print("FC: {0} -> {1}".format(img_size * img_size * 8, img_size * img_size * 2))

    #         # q_values = tflearn.fully_connected(net, actions, activation="linear", name="Q_Values")
    #         # q_values = tf.layers.dense(inputs=net, units=actions, activation="linear", name="Q_Values")
    #         # print("FC: {0} -> {1}".format(img_size * img_size * 2, actions))
    #         # Dont need dueling yet
    #         # v_stream = tflearn.fully_connected(net, ceil(img_size/2), activation="relu")
    #         # v_stream = tflearn.fully_connected(v_stream, 1, activation="linear")

    #         # a_stream = tflearn.fully_connected(net, ceil(img_size/2), activation="relu")
    #         # a_stream = tflearn.fully_connected(a_stream, actions, activation="linear")

    #         # q_values = v_stream + (a_stream - tf.reduce_mean(a_stream, reduction_indices=1, keep_dims=True))

    #         action_index = tf.placeholder(tf.float32, shape=[None, unroll, actions])
    #         target_q = tf.placeholder(tf.float32, shape=[None, unroll, actions])
    #         q_error = action_index * (q_values - target_q)
    #         q_loss = tf.reduce_mean(tf.square(q_error))

    #         variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    #         # Summaries
    #         loss_summary = tf.summary.scalar("Q_Loss", q_loss)
    #         qval_summaries = []
    #         for i in range(actions):
    #             qval_summaries.append(tf.summary.scalar("Action {} QValue".format(i), q_values[0, i]))
    #         average_qval = tf.reduce_mean(q_values)
    #         avg_qvals_summary = tf.summary.scalar("Average QValue", average_qval)
    #         qvals_histogram = tf.summary.histogram("Q Values", q_values)
    #         qval_summaries.append(qvals_histogram)
    #         qval_summaries.append(avg_qvals_summary)
    #         qvals_summary = tf.summary.merge(qval_summaries)

    #     dict = {}
    #     dict["Input"] = inputs
    #     dict["Q_Values"] = q_values
    #     dict["Q_Loss"] = q_loss
    #     dict["Q_Error"] = q_error
    #     dict["Variables"] = variables
    #     dict["Targets"] = target_q
    #     dict["Actions"] = action_index
    #     dict["Loss_Summary"] = loss_summary
    #     dict["QVals_Summary"] = qvals_summary
    #     dict["RNN_State"] = final_lstm_state
    #     dict["RNN_Steps"] = unroll
    #     return dict
