import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import tensorflow as tf
#from tensorflow.python.ops import rnn_cell
#from tensorflow.contrib import rnn

import numpy as np

#import lstmAux as aux

#import cross

import messages as mes

rnn_bias_key = 'rnn/basic_lstm_cell/bias:0'
rnn_kernel_key = 'rnn/basic_lstm_cell/kernel:0'

out_weights_key = 'weights'
out_bias_key = 'biases'

class LSTMRL(nn.Module):


    def __init__(self, input_size, rnn_size, output_size):
        super(LSTMRL, self).__init__()

        self.lstm_layer = nn.LSTM(input_size, rnn_size)
        self.linear_layer = nn.Linear(rnn_size, output_size)

        self.hc_state = (torch.zeros(1,1,rnn_size), torch.zeros(1,1,rnn_size))
        self.hc_state_temp = (torch.zeros(1,1,rnn_size), torch.zeros(1,1,rnn_size))

    def state_update(self, x=None):

        if x:
            self.forward(x)

        self.hc_state = self.hc_state_temp

    def forward(self, x):

        out_rnn, self.hc_state_temp = self.lstm_layer(x, self.hc_state)
        out_linear = self.linear_layer(out_rnn)

        return out_linear[-1][-1]

class LSTM():


    def __init__(self, input_size, rnn_size, output_size, alpha=0.99):

        self.input_size = input_size
        self.rnn_size = rnn_size
        self.output_size = output_size

        self.net = LSTMRL(input_size, rnn_size, output_size)

        self.loss_function = nn.MSELoss(reduction='elementwise-mean')
        self.optimizer = optim.SGD(self.net.parameters(), lr=alpha)

    def __call__(self):
        return self.net.hc_state.detach().numpy()[1]

    def __call__(self, x):

        with torch.no_grad():
            out = self.net(x)

        return out.detach().numpy()

    def state_update(self, x=None):
        self.net.state_update(x)

    def train(self, x, y):

        if not isinstance(x, torch.Tensor):

            x = np.array(x)

            if len(x.shape) < 2:
                x = np.array([x])
            if len(x.shape) < 3:
                x = np.array([x])

            x = torch.Tensor(x)

        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(np.array(y))

        if (len(x.shape) != 3) or x.shape[-1]!=self.input_size:
            raise Exception("Wrong input format")

        if (len(y.shape) != 1) or y.shape[-1]!=self.output_size:
            raise Exception("Wrong target format")

        self.net.zero_grad()

        # model.hidden = model.init_hidden()
        out = self.net(x)
        loss = self.loss_function(out, y)

        loss.backward()
        self.optimizer.step()
