import sys

sys.path.append('../messages/')

sys.path.append('../simulation/simModel/')

sys.path.append('../simulation/simModel/agent/')

sys.path.append('../simulation/simModel/environment/')
sys.path.append('../simulation/simModel/environment/metaModel/')
sys.path.append('../simulation/simModel/environment/metaModel/model/')

sys.path.append('../simulation/GUI/')

import os

import parameters as par

import agent


_defIter = 60
_numRS = 5

_testDirPath = 'tests/test '


def defInput(mes, defVal):
	return int(input("(" + str(defVal) + ") - " + mes) or str(defVal))


testN = 1
while(os.path.exists(_testDirPath + str(testN))):
	testN+=1

os.makedirs(_testDirPath + str(testN))
path = _testDirPath + str(testN) + '/'

fileT = open(path + 'timeP.txt', 'w')
fileR = open(path + 'rewardP.txt', 'w')

a = agent.agent(environment="../simulation/files/maze.txt", graphic=0)

iterations = defInput("Insert number of iterations: ", _defIter)

params = open(path + 'parameters.txt', 'w')

params.write("\nScheduling function:\n")
params.write("\tParameter A value of scheduling function: " + str(a.livePar.scheduleA) + "\n")
params.write("\tParameter B value of scheduling function: " + str(a.livePar.scheduleB) + "\n")
params.write("\tThreshold value of scheduling function: " + str(a.livePar.scheduleThresh) + "\n")

params.write("\nQ learning:\n")
params.write("\tLearning rate value: "+ str(a.livePar.learningRate) + "\n")
params.write("\tDiscount factor value: "+ str(a.livePar.discountFactor) + "\n")
params.write("\tMaximum starting Q value: "+ str(a.livePar.startQMax) + "\n")
params.write("\tMinimum starting Q value: "+ str(a.livePar.startQMin) + "\n")

params.write("\nInterest value update rule\n")
params.write("\tParameter A value of interest update rule: "+ str(a.livePar.interestA) + "\n")
params.write("\tParameter B value of interest update rule: "+ str(a.livePar.interestB) + "\n")
params.write("\tParameter C value of interest update rule: "+ str(a.livePar.interestC) + "\n")
params.write("\tLow bound value of interest update rule: "+ str(a.livePar.iLowBound) + "\n")

params.close()

pT = []
pR = []

for k in range(iterations):

	sPT = []
	sPR = []

	for i in range(_numRS):
		time = 0
		r = par.baseReward
		accumulatedReward = 0

		while(r != par.signalReward):
			a.act(i)

			time+=1
			r = (((a.transitionHistory)[a.time-1])[1])[i]
			accumulatedReward += r

		sPT.append(time)
		sPR.append(accumulatedReward)

		fileT.write("%s\t" % time)
		fileR.write("%s\t" % accumulatedReward)

	pT.append(sPT)
	pR.append(sPR)

	fileT.write("\n")
	fileR.write("\n")

fileT.write("e")
fileR.write("e")

fileT.close()
fileR.close()