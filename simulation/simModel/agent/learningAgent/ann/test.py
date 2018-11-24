import LSTM
import time

x = [1,2,3]

y = [2,3,4,6]

net = LSTM.LSTM(3,5,4)

for i in range(1,10000):

	print("EPOCH #" + str(i), end="")

	start_t = time.time()

	#TRAINING
	net.train_neural_network(x,y)

	#UPDATING STATE
	net.static_update(x)

	end_t = time.time()
	tot_t = round(end_t - start_t, 4)

	print("\ttime: " + str(tot_t) + "s", end="")
	print(" ")

