import sys
import agent


class SysPars():
    def __init__(self):
        self.env = None
        self.loop = True

pars = SysPars()

def defInput(mes, defVal):
    return int(input("(" + str(defVal) + ") - " + mes) or str(defVal))

def runPars(pars,i):
    if (sys.argv)[i] == "path":
        pars.env = (sys.argv)[i+1]
    elif (sys.argv)[i] == "loop":
        pars.loop = True if (sys.argv)[i+1] == "True" else False


if not len(sys.argv)%2:
    print ("wrong argument format")
    sys.exit()

for i in range(1,len(sys.argv)-1,2):
    runPars(pars,i)

print(pars.loop,pars.env)

if pars.env:
    a = agent.agent(environment=pars.env)
else:
    a = agent.agent()

def seqActions():
    nSteps = 500

    while nSteps > 0:
        nSteps = defInput("Insert number of steps: ", nSteps)
        a.nSteps(nSteps,0)

if pars.loop:
    seqActions()