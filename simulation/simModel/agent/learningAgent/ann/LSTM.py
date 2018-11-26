import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import copy

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

    def detach_state(self):
        self.hc_state = (self.hc_state[0].detach(), self.hc_state[1].detach())

    def forward(self, x):

        out_rnn, self.hc_state_temp = self.lstm_layer(x, self.hc_state)
        out_linear = self.linear_layer(out_rnn)

        return out_linear[-1][-1]

class LSTM():

    @classmethod
    def copy_net(cls, net):
        return copy.deepcopy(net)

    def __init__(self, input_size, rnn_size, output_size, alpha=0.99):

        self.input_size = input_size
        self.rnn_size = rnn_size
        self.output_size = output_size

        self.net = LSTMRL(input_size, rnn_size, output_size)

        self.loss_function = nn.MSELoss(reduction='elementwise_mean')
        self.optimizer = optim.SGD(self.net.parameters(), lr=alpha)

    def state(self):
        return self.net.hc_state.detach().numpy()[1]

    def __call__(self, x):

        with torch.no_grad():
            out = self.net(self.toTensor(x))

        return out.detach().numpy()

    def state_update(self, x=None):
        self.net.state_update(self.toTensor(x))

    def train(self, x, y):

        x = self.toTensor(x)

        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(np.array(y))

        if (len(x.shape) != 3) or x.shape[-1]!=self.input_size:
            raise Exception("Wrong input format")

        if (len(y.shape) != 1) or y.shape[-1]!=self.output_size:
            raise Exception("Wrong target format")

        self.net.zero_grad()
        self.net.detach_state()

        out = self.net(x)
        loss = self.loss_function(out, y)

        loss.backward()
        self.optimizer.step()

        return loss

    def toTensor(self, x):

        if not x:
            return x

        if not isinstance(x, torch.Tensor):

            x = np.array(x)

            if len(x.shape) < 2:
                x = np.array([x])
            if len(x.shape) < 3:
                x = np.array([x])

            x = torch.Tensor(x)

        return x

    def duplicate_output(self, idx):

        self.net.linear_layer.weight = F.pad(self.net.linear_layer.weight, (0, 0, 0, 1))
        self.net.linear_layer.bias = F.pad(self.net.linear_layer.bias, (0, 1))

        self.net.linear_layer.weight[-1] += self.net.linear_layer.weight[idx]
        self.net.linear_layer.bias[-1] += self.net.linear_layer.bias[idx]