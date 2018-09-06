import sys
import json

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

import agent

import pip

installed_pkgs = [pkg.key for pkg in pip.get_installed_distributions()]
asPlotPkg = 'matplotlib' in installed_pkgs

if asPlotPkg:
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
    elif (sys.argv)[i] == "v":
        pars.visa = int((sys.argv)[i + 1])
    elif (sys.argv)[i] == "maze":
        pars.mazeName = "maze.txt" if (sys.argv)[i + 1]=="def" else (sys.argv)[i + 1]
    elif (sys.argv)[i] == "pars":
        pars.parsFile = (sys.argv)[i + 1]
    elif (sys.argv)[i] == "origin":
        pars.origin = (sys.argv)[i + 1]

_defIter = 60
_defVisa = 1
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

if not pars.visa :
    pars.visa = defInput("Insert visa value (extra time after goal is found): ", _defVisa)

if not pars.mazeName:
    pars.mazeName = defInput("Insert maze name: ", "maze.txt", string=True)

testN = 1
while (os.path.exists(_testDirPath + '_' + str(testN))):
    testN += 1

for testNIter in range(testN,testN+pars.nExperiments):

    a = agent.agent(environment = "../simulation/files/" + str(pars.mazeName), pars=pars.parsFile, graphic=graphic, suppressPrint=printSupp)

    pT = []
    pR = []
    pU = []

    for k in range(pars.iterations):

        it_desc = " Iteration: %d/%d" % (k+1, pars.iterations)

        time = 0
        r = a.livePar.baseReward
        accumulatedReward = 0

        while (r != a.livePar.goalReward):
            a.act(0)

            time += 1
            r = a.rewardHistory[0]
            accumulatedReward += r

            print(it_desc+"\t #steps: %d\r"%time, end="")

        extra_time = 1
        good_use = 0

        it_desc = it_desc+"\t #steps: %d"%time

        while(extra_time<pars.visa):
            a.act(0)
            extra_time+=1
            r = a.rewardHistory[0]
            if(r == a.livePar.goalReward):
                good_use+=1

            print(it_desc+"\t #visa: %d\r"%extra_time, end="")

        pT.append(time)
        pR.append(accumulatedReward)
        pU.append(good_use/pars.visa)

        a.environment.pullUpAgent()

        if graphic:
            a.environment.graphic.cleanTrack()

        print(" ")


    os.makedirs(_testDirPath + '_' + str(testNIter))
    path = _testDirPath + '_' + str(testNIter) + '/'

    print("Generated test folder: " + path)

    fileT = open(path + 'timeP.txt', 'w')
    fileU = open(path + 'visaUse.txt', 'w')
    fileR = open(path + 'rewardP.txt', 'w')

    print("result files generated")

    if asPlotPkg:
        plt.plot(pT)
        plt.savefig(path + 'timePlot.png')

        plt.clf()

        plt.plot(pU)
        plt.draw()
        plt.savefig(path + 'visaUsePlot.png')

    params = open(path + 'parameters.txt', 'w')

    params.write("Iterations: " + str(pars.iterations) + "\n")
    params.write("Visa: "+ str(pars.visa) + "\n")
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

    params.write("\nSoftmax exploration values:\n")
    params.write("\tsoftmax exp. schedule f. height" + str(a.livePar.height) + "\n")
    params.write("\tsoftmax exp. schedule f. low bound" + str(a.livePar.lowBound) + "\n")
    params.write("\tsoftmax exp. schedule f. speed" + str(a.livePar.speed))
    params.write("\tsoftmax exp. schedule f. starting point" + str(a.livePar.startPoint) + "\n")

    params.close()

    for k in range(pars.iterations):
        fileT.write("%s" % pT[k])
        fileR.write("%s" % pR[k])
        fileU.write("%s" % pU[k])

        if (k < ((pars.iterations) - 1)):
            fileT.write(",")
            fileR.write(",")
            fileU.write(",")

    a.exportPars(path+'pars.JSON')
    json.dump({"time": pT, "rewards": pR, "visa": pU}, open(path+'results.JSON', 'w'))

    fileT.close()
    fileR.close()
    fileU.close()

    del a

path = _testDirPath + '_' + str(testN) + '/'
print("\n\nResults ready from: " + path)
