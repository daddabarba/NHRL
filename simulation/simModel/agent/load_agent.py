import sys
import agent


class SysPars():
    def __init__(self):
        self.env = None
        self.loop = False
        self.GUI = False
        self.noPrint = False

pars = SysPars()

def defInput(mes, defVal):
    return int(input("(" + str(defVal) + ") - " + mes) or str(defVal))

def runPars(pars,i):
    if (sys.argv)[i] == "path":
        pars.env = (sys.argv)[i+1]
    elif (sys.argv)[i] == "loop":
        pars.loop = True if (sys.argv)[i+1] == "True" else False
    elif (sys.argv)[i] == "GUI":
        pars.GUI = True if (sys.argv)[i + 1] == "True" else False
    elif (sys.argv)[i] == "noPrint":
        pars.noPrint = True if (sys.argv)[i + 1] == "True" else False



if not len(sys.argv)%2:
    print ("wrong argument format")
    sys.exit()

for i in range(1, len(sys.argv)-1, 2):
    runPars(pars, i)

if pars.env:
    a = agent.agent(environment=pars.env, graphic=pars.GUI, suppressPrint=pars.noPrint)
else:
    a = agent.agent(graphic=pars.GUI, suppressPrint=pars.noPrint)

def seqActions():
    nSteps = 1

    while nSteps > 0:
        nSteps = defInput("Insert number of steps: ", nSteps)
        a.nSteps(nSteps)

if pars.loop:
    seqActions()