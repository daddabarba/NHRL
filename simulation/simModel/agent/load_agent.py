import sys
import agent

env = None
loop = True

def defInput(mes, defVal):
    return int(input("(" + str(defVal) + ") - " + mes) or str(defVal))

def runPars(i):
    if (sys.argv)[i] == "path":
        env = (sys.argv)[i+1]
    elif (sys.argv)[i] == "loop":
        loop = True if (sys.argv)[i+1] == "True" else False


if not len(sys.argv)%2:
    print "wrong argument format"
    sys.exit()

for i in range(1,len(sys.argv)-1,2):
    runPars(i)

if env:
    a = agent.agent(environment=env)
else:
    a = agent.agent()

def seqActions():
    nSteps = 500

    while nSteps > 0:
        nSteps = defInput("Insert number of steps: ", nSteps)
        a.nSteps(nSteps,0)

if loop:
    seqActions()