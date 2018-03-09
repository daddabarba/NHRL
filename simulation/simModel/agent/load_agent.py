import sys
import agent

def defInput(mes, defVal):
    return int(input("(" + str(defVal) + ") - " + mes) or str(defVal))

if len(sys.argv) != 0:
    a = agent.agent(environment=(sys.argv)[1])
else:
    a = agent.agent()

nSteps = 500

while nSteps > 0:
    nSteps = defInput("Insert number of steps: ", nSteps)
    a.nSteps(nSteps,0)