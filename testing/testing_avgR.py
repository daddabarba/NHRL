import sys
import json

import os

import agent

import qLearning as qLA

import matplotlib.pyplot as plt

graphic = False
printSupp = True

class SysPars():
    def __init__(self):
        self.name = None
        self.nExperiments = None
        self.iterations = None
        self.visa = None
        self.mazeName = None
        self.parsFile = None
        self.origin = None

pars = SysPars()

def runPars(pars,i):
    if (sys.argv)[i] == "name":
        pars.name = (sys.argv)[i+1]
    elif (sys.argv)[i] == "e":
        pars.nExperiments = int((sys.argv)[i+1])
    elif (sys.argv)[i] == "n":
        pars.iterations = int((sys.argv)[i + 1])
    elif (sys.argv)[i] == "maze":
        pars.mazeName = "maze.txt" if (sys.argv)[i + 1]=="def" else (sys.argv)[i + 1]
    elif (sys.argv)[i] == "pars":
        pars.parsFile = (sys.argv)[i + 1]
    elif (sys.argv)[i] == "origin":
        pars.origin = (sys.argv)[i + 1]

_defIter = 1000
_testDirPath = 'tests/'


def defInput(mes, defVal, string=False):
    _in = input("(" + str(defVal) + ") - " + mes) or str(defVal)
    if string:
        return _in

    return int(_in)

for i in range(1,len(sys.argv)-1,2):
    runPars(pars,i)

if pars.origin:
    pars.__dict__.update(json.load(open(pars.origin, 'r')))

if not pars.name:
    pars.name = input("Insert test folder name: ")

dirName = "_".join((pars.name).split())
_testDirPath+=dirName

if not pars.nExperiments:
    pars.nExperiments = defInput("Insert number of experiments: ", 1)

if not pars.iterations :
    pars.iterations = defInput("Insert number of iterations: ", _defIter)

if not pars.mazeName:
    pars.mazeName = defInput("Insert maze name: ", "maze.txt", string=True)

testN = 1
while (os.path.exists(_testDirPath + '_' + str(testN))):
    testN += 1

for testNIter in range(testN,testN+pars.nExperiments):

    a = agent.agent(environment = "../simulation/files/" + str(pars.mazeName), pars=pars.parsFile, graphic=graphic, suppressPrint=printSupp)

    pR = [0.0 for k in range(pars.iterations)]
    pM = [0.0 for k in range(pars.iterations)]

    accumulatedReward = 0
    time = 0

    for k in range(pars.iterations):

        a.act()

        time += 1
        r = a.rewardHistory
        pR[k] = r
        accumulatedReward += pR[k]
        pM[k] = float(accumulatedReward)/time

        it_desc = " time-step: %d/%d - avg reward: %f, total: %f (last reward: %f)" % (k + 1, pars.iterations, pM[k], accumulatedReward, r)
        print(it_desc+"\r", end = "")

    print(" ")

    os.makedirs(_testDirPath + '_' + str(testNIter))
    path = _testDirPath + '_' + str(testNIter) + '/'

    print("Generated test folder: " + path)

    fileM = open(path + 'averageReward.txt', 'w')
    fileR = open(path + 'rewardP.txt', 'w')

    print("result files generated")

    plt.plot(pR)
    plt.savefig(path + 'rewardPlot.png')

    plt.clf()

    plt.plot(pM)
    plt.draw()
    plt.savefig(path + 'avgRewardPlot.png')

    for k in range(pars.iterations):
        fileM.write("%s" % pM[k])
        fileR.write("%s" % pR[k])

        if k < ((pars.iterations) - 1):
            fileR.write(",")
            fileM.write(",")

    a.exportPars(path+'pars.JSON')
    json.dump({"rewards": pR, "avg": pM}, open(path+'results.JSON', 'w'))

    if issubclass(type(a.qAgent), qLA.hierarchy):
        fileTopology = open(path + 'hierarchyTopology.txt', 'w')
        # fileTopology.write(a.qAgent.printHierarchy())
        fileTopology.close()

    fileR.close()
    fileM.close()

    del a

path = _testDirPath + '_' + str(testN) + '/'
print("\n\nResults ready from: " + path)
