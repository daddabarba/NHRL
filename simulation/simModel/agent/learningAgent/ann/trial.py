import LSTM
import numpy as np

import matplotlib.pyplot as plt

MIN_VAL = -500
MAX_VAL = 500

N_EPOCHS = 100
N_EXAMPLES = 10

N_SEQUENCE = 2

IN_SIZE = 4
OUT_SIZE = 2

RNN_SIZE = 100

def f(x):
	return np.array([x[0][0][0]*x[0][0][1],  x[0][0][2]+x[0][0][3]])

net = LSTM.LSTM(IN_SIZE, RNN_SIZE, OUT_SIZE)

X = np.zeros([N_EXAMPLES, N_SEQUENCE, 1, IN_SIZE], dtype=float)
Y = np.zeros([N_EXAMPLES, OUT_SIZE], dtype=float)

# Generate data
for i in range(N_EXAMPLES):

	X[i] = np.random.rand(N_SEQUENCE, 1, IN_SIZE)*(MAX_VAL-MIN_VAL) + MIN_VAL
	Y[i] = f(X[i])

N_TESTS = N_EPOCHS*N_EXAMPLES

losses = np.zeros(N_TESTS)
idx_losses = 0

for i in range(N_EPOCHS):
	for k in range(N_EXAMPLES):

	# print("EPOCH: %d"%i)

		loss = net.train(X[k], Y[k])
		print("\tepoch: %d\tcase: %d\t=> loss: %f"%(i,k,loss))

		losses[idx_losses] = loss
		idx_losses+=1

plt.plot(list(range(N_TESTS)), losses)

plt.ion()
plt.draw()
plt.pause(0.1)