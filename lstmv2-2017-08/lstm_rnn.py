import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

HIDDEN_SIZE = 500
NUM_LAYERS = 2
VOCAB_SIZE = 33
LEARNING_RATE = 0.8
TRAIN_BATCH_SIZE = 10
TRAIN_NUM_STEP = 6

EVAL_BATCH_SIZE = 10
EVAL_NUM_STEP = 6
NUM_EPOCH = 50000
KEEP_PROB = 0.5
# MAX_GRAD_NORM = 5
MAX_GRAD_NORM = 10
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_DECAY = 0.99


class LSTMModel(object):
    def __init__(self, is_training, batch_size, num_steps, batch_counts):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        loss = seq2seq.sequence_loss_by_example([logits],
                                                [tf.reshape(self.targets, [-1])],
                                                [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        self.cost = tf.reduce_sum(loss) / batch_size
        self.global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        # tv = tf.trainable_variables()
        trainable_variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in trainable_variables])
        self.total_loss = self.cost + regularization_cost
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step,
                                                   batch_counts/self.batch_size, LEARNING_RATE_DECAY, staircase=True)

        self.final_state = state

        if not is_training:
            return
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_loss, trainable_variables), MAX_GRAD_NORM)

        # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.total_loss,
                                                                                global_step=self.global_step)
        # self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        with tf.control_dependencies([train_steps, variable_averages_op]):
            self.train_op = tf.no_op(name='train')


def run_epoch(session, model, data, train_op, output_log):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(get_enumerate(data, model.batch_size, model.num_steps)):
        # cost, state, _ = session.run([model.cost, model.final_state, train_op],
        cost, state, _ = session.run([model.total_loss, model.final_state, train_op],
                                     {
                                         model.input_data: x,
                                         model.targets: y,
                                         model.initial_state: state
                                     })

        total_costs += cost
        iters += model.num_steps
        if output_log and step % 20 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / (iters + 0))))

    return np.exp(total_costs / (iters + 0))


def get_enumerate(raw_data, batch_size, num_steps):
    # raw_data = np.array(raw_data, dtype=np.int32)
    # data_len = len(raw_data)
    # batch_len = data_len // batch_size
    # data = np.zeros([batch_size, batch_len], dtype=np.int32)
    # for i in range(batch_size):
    #     print(i)
    #     data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    #
    # epoch_size = (batch_len - 1) // num_steps
    # if epoch_size == 0:
    #     raise ValueError("epoch_size = 0, decreae batch_size or num_steps")
    # for i in epoch_size:
    #     x = data[:, i * num_steps:(i+1)*num_steps]
    #     y = data[:, i * num_steps + 1: (i+1)*num_steps + 1]
    #     yield (x,y)
    raw_data = np.array(raw_data)
    data_len = len(raw_data)
    epoch_size = data_len // batch_size
    # print(epoch_size, data_len, data_len // batch_size * batch_size)
    if data_len - data_len // batch_size * batch_size <= 0:
        raise ValueError("data len should be greater at least 1")
    for i in range(data_len // batch_size):
        x = raw_data[i * batch_size:(i + 1) * batch_size, :]
        y = raw_data[i * batch_size + 1:(i + 1) * batch_size + 1, :]
        yield (x, y)


def get_data():
    lottery_file = "train_data.csv"
    lotteries = []
    with open(lottery_file, "r") as f:
        for one_term in f:
            one_term_digits_string = one_term.strip().split(",")
            one_term_digits = [int(num) - 1 for num in one_term_digits_string]
            lotteries.append(one_term_digits)
    # print(lotteries)
    train_data = lotteries[0:2001]
    valid_data = lotteries[2000:2101]
    eval_data = lotteries[2100:2141]

    return train_data, valid_data, eval_data


def main():
    train_data, valid_data, test_data = get_data()
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("lottery_train", reuse=None, initializer=initializer):
        train_model = LSTMModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP, len(train_data))

    with tf.variable_scope("lottery_train", reuse=True, initializer=initializer):
        eval_model = LSTMModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP, len(valid_data))
    saver = tf.train.Saver()
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_data, train_model.train_op, True)

            valid_perplexity = run_epoch(session, eval_model, valid_data, tf.no_op(), False)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))
            if valid_perplexity < 10 or (i >= 150 and i % 50 == 0):
                saver.save(session, "./model.ckpt")

        test_perplexity = run_epoch(session, eval_model, test_data, tf.no_op(), False)
        print("Test Perplexity: %.3f" % test_perplexity)


main()
