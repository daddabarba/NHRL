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

class State_Set():

    def __init__(self, net, newState):

        self.net = net

        self.oldState = net.net.hc_state
        self.oldState_temp = net.net.hc_state_temp

        self.newState = newState

    def __enter__(self):

        self.net.net.hc_state = self.newState

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.net.net.hc_state = self.oldState
        self.net.net.hc_state_temp = self.oldState_temp

class LSTMRL(nn.Module):

    def __init__(self, input_size, rnn_size, output_size, linear_layer=None, lstm_layer=None, hc_state=None):
        super(LSTMRL, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size
        self.output_size = output_size

        if lstm_layer is None:
            self.lstm_layer = nn.LSTM(input_size, rnn_size)
        else:
            self.lstm_layer = lstm_layer

        if linear_layer is None:
            self.linear_layer = nn.Linear(rnn_size, output_size)
        else:
            self.linear_layer = linear_layer

        if hc_state is None:
            self.hc_state = (torch.zeros(1, 1, rnn_size), torch.zeros(1, 1, rnn_size))
        else:
            self.hc_state = hc_state

        self.hc_state_temp = (torch.zeros(1, 1, rnn_size), torch.zeros(1, 1, rnn_size))

    def __deepcopy__(self, memodict={}):

        return LSTMRL(self.input_size, self.rnn_size, self.output_size, copy.deepcopy(self.linear_layer), copy.deepcopy(self.lstm_layer), copy.deepcopy(self.hc_state))

    def state_update(self, x=None):

        if x is not None:
            self.forward(x)

        self.hc_state = self.hc_state_temp

    def detach_state(self):
        self.hc_state = (self.hc_state[0].detach(), self.hc_state[1].detach())

    def forward(self, x):

        out_rnn, self.hc_state_temp = self.lstm_layer(x, self.hc_state)
        out_linear = self.linear_layer(out_rnn)

        return out_linear[-1][-1]

class LSTM():

    def __init__(self, input_size, rnn_size, output_size, alpha=0.99, net=None):

        self.input_size = input_size
        self.rnn_size = rnn_size
        self.output_size = output_size

        self.alpha = alpha

        if net is None:
            self.net = LSTMRL(input_size, rnn_size, output_size)
        else:
            self.net = net

        self.loss_function = nn.MSELoss(reduction='elementwise_mean')
        self.optimizer = optim.SGD(self.net.parameters(), lr=alpha)

    def __call__(self, x):

        with torch.no_grad():
            out = self.net(self.toTensor(x))

        return out.detach().numpy()

    def __deepcopy__(self, memodict={}):
        return copy.copy(self)

    def __copy__(self):
        return LSTM(self.input_size, self.rnn_size, self.output_size, self.alpha, copy.deepcopy(self.net))

    def state(self, i=1):
        return self.net.hc_state[i].detach().numpy()[0][0]

    def hcState(self):
        return self.net.hc_state

    def state_update(self, x=None):
        self.net.state_update(self.toTensor(x))

    def train(self, x, y):

        x = self.toTensor(x)

        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(y)

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

        if (not isinstance(x, np.ndarray) and not isinstance(x, torch.Tensor)) and x is None:
            return x

        if not isinstance(x, torch.Tensor):

            x = np.array(x)

            if len(x.shape) < 2:
                x = np.array([x])
            if len(x.shape) < 3:
                x = np.array([x])

            x = torch.Tensor(x)

        return x

    def getMlp(self):
        return self.net.linear_layer.weight.detach().numpy(), self.net.linear_layer.bias.detach().numpy()

    def setMlp(self, newW, newB):

        newW = torch.nn.Parameter(torch.Tensor(newW), requires_grad=True)
        newB = torch.nn.Parameter(torch.Tensor(newB), requires_grad=True)

        if (newW.shape[0] != newB.shape[0]) or (newW.shape[1] != self.rnn_size):
            return

        self.net.linear_layer.weight = newW
        self.net.linear_layer.bias = newB

    def duplicate_output(self, idx):

        self.net.linear_layer.weight = torch.nn.Parameter(F.pad(self.net.linear_layer.weight, (0, 0, 0, 1)))
        self.net.linear_layer.bias = torch.nn.Parameter(F.pad(self.net.linear_layer.bias, (0, 1)))

        self.net.linear_layer.weight[-1] += self.net.linear_layer.weight[idx]
        self.net.linear_layer.bias[-1] += self.net.linear_layer.bias[idx]