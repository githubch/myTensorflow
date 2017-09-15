import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

HIDDEN_SIZE = 256
NUM_LAYERS = 4
VOCAB_SIZE = 33
LEARNING_RATE = 0.002
TRAIN_BATCH_SIZE = 10
TRAIN_NUM_STEP = 6 * 10

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 6 * 10
NUM_EPOCH = 20000
KEEP_PROB = 0.5
# MAX_GRAD_NORM = 5
MAX_GRAD_NORM = 5
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_DECAY = 0.99

TERM_SIZE = 6

class LSTMModel(object):
    def __init__(self, is_training, batch_size, num_steps, batch_counts):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, TERM_SIZE])

        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=KEEP_PROB)
        rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * NUM_LAYERS)

        self.initial_state_fw = rnn_cell_fw.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)
        with tf.variable_scope('bidirectional_rnn'):
            outputs_fw = []
            state_fw = self.initial_state_fw
            with tf.variable_scope("rnn_fw"):
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    cell_output_fw, state_fw = rnn_cell_fw(inputs[:, time_step, :], state_fw)
                    if time_step >= num_steps - TERM_SIZE:
                        outputs_fw.append(cell_output_fw)
            output = tf.reshape(tf.concat(outputs_fw, 1), [-1, HIDDEN_SIZE])
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        loss = seq2seq.sequence_loss_by_example([logits],
                                                [tf.reshape(self.targets, [-1])],
                                                weights=[tf.ones([batch_size * TERM_SIZE], dtype=tf.float32)])

        self.cost = tf.reduce_mean(loss)
        self.final_state_fw = state_fw
        self.predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        self.correct_prediction = tf.equal(self.predictions, tf.reshape(self.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step, batch_counts / self.batch_size,
                                                        LEARNING_RATE_DECAY, staircase=True)
        trainable_variables = tf.trainable_variables()
        # regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in trainable_variables])
        regularization_cost = tf.nn.l2_loss(weight) + tf.nn.l2_loss(bias)
        self.cost = self.cost + REGULARIZATION_RATE * regularization_cost
        if not is_training:
            return
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, global_step=self.global_step)
        # self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables),
        #                                           global_step=tf.train.get_or_create_global_step())


def run_epoch(session, model, data, train_op, output_log, total_iter, is_training):
    total_costs = 0.0
    iters = 0
    data_len = len(data) // model.batch_size
    for step, (x, y) in enumerate(get_enumerate_in_list(data, model.batch_size, model.num_steps)):
        state_fw = session.run(model.initial_state_fw)
        cost, state_fw, accuracy, learning_rate, predications, _ = session.run(
            [model.cost, model.final_state_fw,
             model.accuracy, model.learning_rate, model.predictions, train_op],
            {
                model.input_data: x,
                model.targets: y,
                model.initial_state_fw: state_fw,
            })

        total_costs += cost
        iters += model.num_steps
        if output_log and (step == 0 or step == data_len // 5 or step == data_len * 2 // 5 or step == data_len * 3 // 5
                           or step == data_len * 4 // 5 or step == data_len - 1):
            print("After %d steps, perplexity is %.3f, cost is %0.6f" % (step, np.exp(total_costs / (iters + 0)), cost))
            print("Accuracy = %.6f" % accuracy)
            print("Learning rate = %.10f" % learning_rate)
            print("Predications is as following:")
            for i in range(5):
                print(predications[i * 6: (i + 1) * 6])
            print("Labels is as following:")
            print(y[0:5])
    return np.exp(total_costs / (iters + 0)), accuracy


def get_enumerate_in_array(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data)
    data_len = len(raw_data) - batch_size
    # if data_len - data_len // batch_size * batch_size <= 0:
    #    raise ValueError("data len should be greater at least 1")
    for i in range(data_len // batch_size):
        x = raw_data[i * batch_size:(i + 1) * batch_size, :]
        y = raw_data[i * batch_size + 1:(i + 1) * batch_size + 1, :]
        yield (x, y)


def get_data_in_array():
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


def get_enumerate_in_list(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - num_steps - TERM_SIZE) // TERM_SIZE + 1
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * TERM_SIZE:i * TERM_SIZE + num_steps]
        y = data[:, i * TERM_SIZE + num_steps:(i + 1) * TERM_SIZE + num_steps]
        yield (x, y)


def get_data_in_list():
    lottery_file = "train_data.csv"
    lotteries = []
    with open(lottery_file, "r") as f:
        for one_term in f:
            one_term_digits_string = one_term.strip().split(",")
            [lotteries.append(int(num) - 1) for num in one_term_digits_string]
    train_data = lotteries[0:12660]
    valid_data = lotteries[12600: 12600 + 102]
    eval_data = lotteries[12600 + 102: 12600 + 102 + 84]

    return train_data, valid_data, eval_data


def main():
    train_data, valid_data, test_data = get_data_in_list()
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("lottery_train", reuse=None, initializer=initializer):
        train_model = LSTMModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP, len(train_data)*3)

    with tf.variable_scope("lottery_train", reuse=True, initializer=initializer):
        eval_model = LSTMModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP, len(valid_data))
    saver = tf.train.Saver()
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_data, train_model.train_op, True, i, True)

            valid_perplexity, accuracy = run_epoch(session, eval_model, valid_data, tf.no_op(), False, i, False)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))
            if (i >= 500 and i % 500 == 0) or (valid_perplexity < 10 and i % 20 == 0) or (
                    accuracy > 0.9 and i % 20 == 0):
                saver.save(session, "./model.ckpt")

        test_perplexity, accuracy = run_epoch(session, eval_model, test_data, tf.no_op(), False, 0, False)
        print("Test Perplexity: %.3f" % test_perplexity)
        print("Test Accuracy: %.3f" % accuracy)

main()
