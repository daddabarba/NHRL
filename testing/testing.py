import sys

sys.path.append('../messages/')

sys.path.append('../simulation/simModel/')

sys.path.append('../simulation/simModel/agent/')
sys.path.append('../simulation/simModel/agent/learningAgent/')
sys.path.append('../simulation/simModel/agent/sensors/')

sys.path.append('../simulation/simModel/environment/')
sys.path.append('../simulation/simModel/environment/features')
sys.path.append('../simulation/simModel/environment/metaModel/')
sys.path.append('../simulation/simModel/environment/metaModel/model/')
sys.path.append('../systemFunctions/')

sys.path.append('../simulation/GUI/')

import os

import matplotlib.pyplot as plt

import agent

_defIter = 60
_defVisa = 1
_testDirPath = 'tests/test '


def defInput(mes, defVal):
    return int(input("(" + str(defVal) + ") - " + mes) or str(defVal))


testN = 1
while (os.path.exists(_testDirPath + str(testN))):
    testN += 1
os.makedirs(_testDirPath + str(testN))
path = _testDirPath + str(testN) + '/'

fileT = open(path + 'timeP.txt', 'w')
fileU = open(path + 'visaUse.txt', 'w')
fileR = open(path + 'rewardP.txt', 'w')

a = agent.agent(environment = "../simulation/files/maze.txt" ,graphic=0)

iterations = defInput("Insert number of iterations: ", _defIter)
visa = defInput("Insert visa value (extra time after goal is found): ", _defVisa)

params = open(path + 'parameters.txt', 'w')

params.write("Iterations: " + str(iterations) + "\n")
params.write("Visa: "+ str(visa) + "\n")
params.write("Maze Size: " + str(a.environment.size[0]) + "*" + str(a.environment.size[1]) + "\n\n")

params.write("\nReward values:")
params.write("\tbase reward: " + str(a.livePar.baseReward))
params.write("\tgoal reward: " + str(a.livePar.goalReward))

params.write("\nScheduling function:\n")
params.write("\tParameter A value of scheduling function: " + str(a.livePar.scheduleA) + "\n")
params.write("\tParameter B value of scheduling function: " + str(a.livePar.scheduleB) + "\n")
params.write("\tThreshold value of scheduling function: " + str(a.livePar.scheduleThresh) + "\n")

params.write("\nQ learning:\n")
params.write("\tLearning rate value: " + str(a.livePar.learningRate) + "\n")
params.write("\tDiscount factor value: " + str(a.livePar.discountFactor) + "\n")
params.write("\tMaximum starting Q value: " + str(a.livePar.startQMax) + "\n")
params.write("\tMinimum starting Q value: " + str(a.livePar.startQMin) + "\n")

params.close()

pT = []
pR = []
pU = []

for k in range(iterations):
    time = 0
    r = a.livePar.baseReward
    accumulatedReward = 0

    while (r != a.livePar.goalReward):
        a.act(0)

        time += 1
        r = (((a.transitionHistory)[a.time - 1])[1])[0]
        accumulatedReward += r

    extra_time = 1
    good_use = 0
    while(extra_time<visa):
        a.act(0)
        extra_time+=1
        r = (((a.transitionHistory)[a.time - 1])[1])[0]
        if(r == a.livePar.goalReward):
            good_use+=1

    fileT.write("%s" % time)
    fileR.write("%s" % accumulatedReward)

    if(k<((iterations)-1)):
        fileT.write(",")
        fileR.write(",")

    pT.append(time)
    pR.append(accumulatedReward)
    pU.append(good_use/visa)

    a.environment.pullUpAgent()

#fileT.write("end")
#fileR.write("end")

fileT.close()
fileR.close()
fileU.close

plt.plot(pT)
plt.savefig(path + 'timePlot.png')

plt.clf()

plt.plot(pU)
plt.savefig(path + 'visaUsePlot.png')
