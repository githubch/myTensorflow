import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy

ticket_file = "data.csv"
tickets = []
with open(ticket_file, "r") as f:
    for one_term in f:
        one_term_digits_string = one_term.strip().split(",")
        one_term_digits = [int(num) for num in one_term_digits_string]
        tickets.append(one_term_digits)
        # print tickets

to_num = lambda digit: digit - 1;
term_digits_vector = [list(map(to_num, term_digits)) for term_digits in tickets]
# should remove last collomn about bule digit
digits_range = 33
time_terms = 18
term_size = 6
batch_size = time_terms
n_chunk = len(term_digits_vector)
print n_chunk
label_size = 1

x_inputs = []
y_labels = []
for i in range(n_chunk - label_size - batch_size):
    start_index = i

    end_index = start_index + batch_size
    batches = term_digits_vector[start_index:end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), 40, np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    x_inputs.append(xdata)

    ydata = np.full((label_size, length), 40, np.int32)
    labels = term_digits_vector[end_index:end_index + label_size]
    for row in range(label_size):
        ydata[row, :len(labels[row])] = labels[row]
    y_labels.append(ydata)

# ----------------RNN------------

# input_data = tf.placeholder(tf.int32, [batch_size, None])
input_data = tf.placeholder(tf.int32, [time_terms, term_size])
# output_targets = tf.placeholder(tf.float32, [batch_size, None, digits_range])
output_targets = tf.placeholder(tf.int32, [label_size, None])


# Define RNN
def recurrent_neural_network(model='lstm', num_units=128, num_layers=4):
    rnn = core_rnn_cell_impl
    if model == 'lstm_m':
        cell = rnn.LSTMCell(num_units, state_is_tuple=True, use_peepholes=True, num_proj=num_units)
    elif model == 'lstm':
        cell = rnn.BasicLSTMCell(num_units, state_is_tuple = True)

    cell = rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    # cell = core_rnn_cell_impl.DropoutWrapper(cell, output_keep_prob=0.7)

    initial_state = cell.zero_state(term_size, tf.float32)

    with tf.variable_scope('rnnlm_now'):
        softmax_w = tf.get_variable("softmax_w", [num_units * time_terms, digits_range])  # 1-33
        softmax_b = tf.get_variable("softmax_b", [digits_range])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [digits_range, num_units])
            inputs = tf.nn.embedding_lookup(embedding, input_data)
            outputs, last_state = tf.nn.dynamic_rnn(cell, inputs,
                                                    initial_state=initial_state, scope='rnnlm_now',
                                                    time_major=True)
            output = tf.reshape(tf.concat(outputs, 1), [term_size, -1])
            output = tf.nn.dropout(output, 0.5)
            logits = tf.matmul(output, softmax_w) + softmax_b
            probs = tf.nn.softmax(logits)

    return logits, last_state, probs, cell, initial_state


def train_neural_network():
    logits, last_state, _, _, _ = recurrent_neural_network()
    # targets = tf.reshape(output_targets, [-1, digits_range])
    targets = tf.reshape(output_targets, [-1])

    loss = seq2seq.sequence_loss_by_example(logits=[logits], targets=[targets],
                                            weights=[tf.ones_like(targets, dtype=tf.float32)])
                                            # softmax_loss_function=softmax_cross_entropy)
    print(logits.get_shape())
    print(targets.get_shape())
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 6)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # train_op = optimizer.apply_gradients(zip(grads, tvars))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        lr = 0.1
        lr_decay = 0.002
        for epoch in range(1000):
            mini_lr = lr_decay * (0.97 ** (epoch * 1.0 / 1001))
            lr = lr * 1.0 / 10
            if lr < mini_lr:
                lr = mini_lr
            if epoch > 0 and epoch % 55 == 0:
                lr_decay /= 10.0
            sess.run(tf.assign(learning_rate, lr))
            n = 0
            batches = n_chunk - label_size - batch_size
            for batche in range(batches):
                train_loss, _, _ = sess.run([cost, last_state, train_op],
                                            feed_dict={input_data: x_inputs[n], output_targets: y_labels[n]})
                n += 1
                if n == batches / 4 or n == batches * 2 / 4 or n == batches * 3 / 4 \
                        or n == 1 or n == batches - 1:
                    print(epoch, batche, train_loss)
                    print lr
                if epoch > 34 and epoch % 7 == 0 and (n == batches / 4 or n == batches * 2 / 4 or n == batches * 3 / 4 \
                        or n == 1 or n == batches - 1):
                    saver.save(sess, 'ticket.module', global_step=epoch)


train_neural_network()
