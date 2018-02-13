import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn

import numpy as np


# import rnnPar as par

class LSTM():
    def __init__(self, input_size, rnn_size, output_size):
        # setting hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = rnn_size

        # counting epochs
        self.epoch = 1

        # i/o placeholders
        self.xPH = tf.placeholder('float', [None, self.input_size])
        self.yPH = tf.placeholder('float')

        # time series of inputs (and outputs)
        self.input_baches = []
        self.target_baches = []

        # saving lstm prediciton and state function (w.r.t. input placeholder)
        (self.prediction, self.state) = self.neural_network_model(self.xPH)
        # setting cost function (in function of prediction and output placeholder for target values)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.yPH))
        # setting optimizer
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # starting session
        self.sess = tf.Session()
        (self.sess).run(tf.global_variables_initializer())

    # reshapres input tensors to the correct format
    def reshapeData(self, x):
        # matrix transpose
        x = tf.transpose(x, [1, 0])
        # split in timesteps
        x = tf.reshape(x, [-1, self.input_size])
        x = tf.split(x, self.epoch)

        return x

        # def convertData(self, xs):
        # return np.vstack([np.expand_dims(x, 0) for x in xs])

    # LSTM ANN function
    def neural_network_model(self, x):
        x = self.reshapeData(x)

        lstm_layer = rnn_cell.BasicLSTMCell(self.rnn_size)
        outputs, states = rnn.static_rnn(lstm_layer, x, dtype=tf.float32, sequence_length=[1])

        output_layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.output_size])),
                        'biases': tf.Variable(tf.random_normal([self.output_size]))}

        output = tf.matmul(outputs[-1], output_layer['weights']) + output_layer['biases']

        return (output, states[-1])

    def train_neural_network(self, train_x, train_y):
        # train_x = tf.transpose(train_x,[1,0])
        # train_x = tf.reshape(train_x,[-1, par.input_size])
        # train_x = tf.split(train_x, par.t)

        # train_x = train_x.eval(session=sess)

        self.input_baches.append(train_x)
        self.target_baches.append(train_y)

        fd = {self.xPH: np.array(self.input_baches), self.yPH: np.array(self.target_baches)}

        prediction, state, _ = (self.sess).run([self.prediction, self.state, self.optimizer], feed_dict=fd)
        # epoch_loss += c

        self.epoch += 1
        return (prediction, state)

    def predict(self, input=None):
        feed_dict = {self.xPH: np.array(self.input_baches + ([input] if input != None else []))}

        return (self.sess).run([self.prediction], feed_dict)

    def getLasPrediction(self, input=None):
        return (self.predict(input))[-1][-1]

    def getState(self):
        return self.getTensor(self.state)

    def getTensor(self, T):
        feed_dict = {self.xPH: np.array(self.input_baches)}

        return self.sess.run(T, feed_dict)