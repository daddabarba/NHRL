import sys

sys.path.append('../../../../../messages/')

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn

import numpy as np

import lstmAux as aux

import cross

import messages as mes

rnn_bias_key = 'rnn/basic_lstm_cell/bias:0'
rnn_kernel_key = 'rnn/basic_lstm_cell/kernel:0'

out_weights_key = 'weights'
out_bias_key = 'biases'


class LSTM():
    def restart(self, input_size, rnn_size, output_size, alpha=-1, session=None, scope="lstm"):
        return self.__class__(input_size, rnn_size, output_size, alpha, session, scope, self.cell, self.train_cell, self.batch)

    def __init__(self, input_size, rnn_size, output_size, alpha=-1, session=None, scope="lstm", cell=None, train_cell=None, batch=True):
        # storing scope name
        self.scope = aux.uniqueScope(scope)

        self.alpha = alpha
        self.batch = batch

        # setting hyperparameters
        self.input_size = input_size
        self.output_size = output_size if type(output_size) != type({}) else len(output_size[out_bias_key])
        self.rnn_size = rnn_size if type(rnn_size) != type({}) else int(len(rnn_size[rnn_bias_key]) / 4)

        # counting epochs
        self.epoch = 1

        # i/o placeholders
        self.xPH = tf.placeholder('float', shape=(None, self.input_size))
        self.yPH = tf.placeholder('float', shape=(1, self.output_size))

        with tf.variable_scope(self.scope):

            # saving lstm prediciton and state function (w.r.t. input placeholder)
            self.prediction = self.neural_network_model()
            # setting cost function (in function of prediction and output placeholder for target values)
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=self.yPH))
            # setting optimizer
            if alpha and alpha>0:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.cost)
            else:
                self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # starting session
        if not session:
            session = tf.Session()

        self.sess = session
        (self.sess).run(tf.global_variables_initializer())

        if type(output_size) == type({}):
            self.override(self.output_layer[out_weights_key], output_size[out_weights_key])
            self.override(self.output_layer[out_bias_key], output_size[out_bias_key])

        if type(rnn_size) == type({}):
            rnn_size = dict([(self.scope + '/' + name, val) for (name, val) in rnn_size.items()])

            for v in tf.global_variables():
                if (v.name) in list(rnn_size.keys()):
                    self.override(v, rnn_size[v.name])

        if cell:
            self.sess.run(self.cell.c.assign(cell.c))
            self.sess.run(self.cell.h.assign(cell.h))

        if train_cell:
            self.sess.run(self.train_cell.c.assign(train_cell.c))
            self.sess.run(self.train_cell.h.assign(train_cell.h))


    # reshapres input tensors to the correct format
    def reshapeData(self, x):
        # matrix transpose
        x = tf.transpose(x, [1, 0])
        # split in time-steps
        x = tf.reshape(x, [-1, self.input_size])
        x = tf.split(x, 1)

        return x

        # def convertData(self, xs):
        # return np.vstack([np.expand_dims(x, 0) for x in xs])

    # LSTM ANN function
    def neural_network_model(self):
        x = self.reshapeData(self.xPH)

        self.lstm_layer = rnn_cell.BasicLSTMCell(self.rnn_size)

        state_vars = self.lstm_layer.zero_state(1, dtype=tf.float32)

        c = tf.Variable(state_vars.c, trainable=False)
        h = tf.Variable(state_vars.h, trainable=False)

        self.cell = tf.contrib.rnn.LSTMStateTuple(c, h)
        self.train_cell = tf.contrib.rnn.LSTMStateTuple(tf.Variable(c), tf.Variable(h))

        self.outputs, self.state = rnn.static_rnn(self.lstm_layer, x, dtype=tf.float32, sequence_length=[1], initial_state=self.cell)

        W = tf.Variable(tf.random_normal([self.rnn_size, self.output_size]))
        b = tf.Variable(tf.random_normal([self.output_size]))

        self.output_layer = {out_weights_key: W, out_bias_key: b}

        return tf.matmul(self.outputs[-1], self.output_layer[out_weights_key]) + self.output_layer[out_bias_key]

    def static_update(self, x):
        self.set_state(self.getFullState(x))

    def set_state(self, tuple):
        aux.assignRNNTuple(self.sess, self.cell, tuple)
        self.batch = True

    def set_train_state(self, tuple):
        aux.assignRNNTuple(self.sess, self.train_cell, tuple)

    def train_neural_network(self, train_x, train_y, state=None):

        if not state:
            if self.batch:
                state = self.train_cell
            else:
                state = self.cell

        with aux.tempAssign(self.sess, self.cell, state):

            fd = {self.xPH: np.array([train_x]), self.yPH: [train_y]}

            prediction, _, c = (self.sess).run([self.prediction, self.optimizer, self.cost], feed_dict=fd)
            mes.currentMessage("Epoch loss: " + str(c))

            self.epoch += 1

            self.set_train_state(self.getFullState(train_x))
            self.set_state(self.train_cell)

            return (prediction)

    def predict(self, x):
        fd = {self.xPH: np.array([x])}
        return (self.sess).run([self.prediction], feed_dict=fd)

    def getLastPrediction(self, x):
        return (self.predict(x))[-1][-1]

    def getFullState(self, x):
        fd = {self.xPH: np.array([x])}
        return (self.sess).run([self.state], feed_dict=fd)[-1]

    def getLastState(self, x=None):

        if not x:
            return self.sess.run(self.cell.c)

        fd = {self.xPH: np.array([x])}
        return (self.sess).run([self.state.c], feed_dict=fd)[-1]

    def copyOutput(self, ind):
        new_shape = list(self.sess.run(tf.shape(self.output_layer[out_weights_key])) + [0, 1])

        _w = np.transpose(self.sess.run(self.output_layer[out_weights_key]))[ind]
        _w = self.sess.run(tf.reshape(_w, [len(_w), 1]))

        new_W = self.changeShape(self.output_layer[out_weights_key], newShape=new_shape)
        new_W = self.override(new_W, _w, at=[0, new_shape[1] - 1])

        _b = self.sess.run(self.output_layer[out_bias_key])[ind]

        new_b = self.changeShape(self.output_layer[out_bias_key],  newShape=[new_shape[1]])
        new_b = self.override(new_b, [_b - np.log(2)], [new_shape[1] - 1])
        new_b = self.override(new_b, [_b - np.log(2)], [ind])

        output_layer = {out_weights_key: self.sess.run(new_W), out_bias_key: self.sess.run(new_b)}
        rnn_layer = self.getCopy()['rnn']

        return self.restart(self.input_size, rnn_layer, output_layer, self.alpha, self.sess, self.scope)

    def copyNetwork(self):
        parameters = self.getCopy()

        return self.restart(self.input_size, parameters['rnn'], parameters['out'], self.alpha, self.sess, self.scope)

    def getCopy(self):

        W = self.sess.run(self.output_layer[out_weights_key])
        b = self.sess.run(self.output_layer[out_bias_key])

        output_layer = {out_weights_key: W, out_bias_key: b}

        for v in tf.global_variables():
            if v.name == self.scope+'/'+rnn_bias_key:
                rnn_bias = self.sess.run(v)
            if v.name == self.scope+'/'+rnn_kernel_key:
                rnn_kernel = self.sess.run(v)

        rnn_layer = {rnn_kernel_key:rnn_kernel, rnn_bias_key: rnn_bias}

        return {'out': output_layer, 'rnn':rnn_layer}

    def changeShape(self, tensor, newShape):
        new = tf.Variable(tf.zeros(newShape))
        self.sess.run(tf.variables_initializer([new]))

        old = np.ndarray.tolist(self.sess.run(tensor))

        new = self.override(new, old)
        return new

    def override(self, tensor, values, at=None):
        shape_tensor = list(self.sess.run(tf.shape(tensor)))
        shape_values = list(self.sess.run(tf.shape(values)))

        if not at:
            at = np.ndarray.tolist(np.zeros(len(shape_values), dtype=int))

        sup = [range(at[i], at[i] + shape_values[i]) for i in range(len(at))]

        indices = cross.crossProduct(sup)

        val = [self.sess.run(tensor[i]) for i in indices]
        values = list(self.sess.run(tf.reshape(values, [-1])))

        delta1 = tf.sparse_tensor_to_dense(tf.SparseTensor(indices, val, shape_tensor))
        delta2 = tf.sparse_tensor_to_dense(tf.SparseTensor(indices, values, shape_tensor))

        result = tensor - delta1 + delta2
        self.sess.run(result)

        self.sess.run(tensor.assign(result))

        return tensor

    def getVars(self):
        return [v for v in tf.global_variables() if v.name.startswith(self.scope+'/')]

    def __del__(self):
        self.sess.close()
