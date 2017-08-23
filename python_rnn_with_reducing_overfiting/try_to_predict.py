import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

ticket_file = "test.csv"
#ticket_file = "data.csv"
tickets = []
with open(ticket_file, "r") as f:
    for one_term in f:
        one_term_digits_string = one_term.strip().split(",")
        one_term_digits = [int(num) for num in one_term_digits_string]
        tickets.append(one_term_digits)
        # print tickets

to_num = lambda digit: digit - 1;
term_digits_vector = [list(map(to_num, term_digits)) for term_digits in tickets]

digits_range = 33
time_terms = 18
term_size = 6
batch_size = time_terms
n_chunk = len(term_digits_vector)
label_size = 1

x_inputs = []
for i in range(n_chunk - batch_size + 1):
    start_index = i

    end_index = start_index + batch_size
    batches = term_digits_vector[start_index:end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), 40, np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    x_inputs.append(xdata)

# input_data = tf.placeholder(tf.int32, [batch_size, None])
input_data = tf.placeholder(tf.int32, [time_terms, term_size])


# Define RNN
def recurrent_neural_network(model='lstm', num_units=128, num_layers=4):
    rnn = core_rnn_cell_impl
    if model == 'lstm_m':
        cell = rnn.LSTMCell(num_units, state_is_tuple=True, use_peepholes=True, num_proj=num_units)
    elif model == 'lstm':
        cell = rnn.BasicLSTMCell(num_units, state_is_tuple=True)

    cell = rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    #cell = core_rnn_cell_impl.DropoutWrapper(cell, output_keep_prob=0.6)

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
            logits = tf.matmul(output, softmax_w) + softmax_b
            probs = tf.nn.softmax(logits)

    return logits, last_state, probs, cell, initial_state


def gen_poetry():
    _, last_state, probs, cell, initial_state = recurrent_neural_network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, './ticket.module-154')
        for i in range(len(x_inputs)):
            x = x_inputs[i]
            print 100 * '='
            if i < len(x_inputs) - 1:
                print "test labels: "
                print map(lambda x: x + 1, x_inputs[i + 1][-1])
            state_ = sess.run(cell.zero_state(term_size, tf.float32))
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            print "test prediction:"
            print map(lambda x: x + 1, np.argmax(probs_, axis=1))


gen_poetry()
