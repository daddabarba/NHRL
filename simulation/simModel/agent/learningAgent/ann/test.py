import LSTM
import time

x = [1,2,3]

y = [2,3,4,6]

net = LSTM.LSTM(3,5,4)

for i in range(1,10000):

	start_t = time.time()

	#TRAINING
	cost = net.train(x,y)

	#UPDATING STATE
	net.state_update()

	end_t = time.time()
	tot_t = round(end_t - start_t, 4)

	print("EPOCH #" + str(i) + "\ttime: " + str(tot_t) + "s" + "\tloss: " + str(cost.detach().numpy()))

