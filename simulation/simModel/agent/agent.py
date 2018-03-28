import sys

sys.path.append('../environment/')
sys.path.append('../agent/learningAgent/')
sys.path.append('../agent/sensors/')

sys.path.append('../../../systemFunctions/')

sys.path.append('../../../messages/')

import environment as env
import qLearningAgent as qLA
import sensors

import agentPar as par

import metaFunctions as meta

import messages as mes


def attachSensors():
    return meta.getFunctionsDefinitions(sensors)

class agent:
    def __init__(self, startingState="c", environment="../../files/maze.txt", graphic=1, suppressPrint = False):

        mes.suppress = suppressPrint

        mes.currentMessage("sensors")
        (self.sensors, self.sensorsNames) = attachSensors()

        mes.currentMessage("environment")
        self.environment = env.environment(environment, self, startingState, graphic)

        mes.settingMessage("live parameters")
        self.livePar = par.agentPar()
        mes.setMessage("live parameters")


        self.problemStateDefinition = ["leftWall", "rightWall", "topWall", "bottomWall", "previousAction"]
        self.goalStateDefinition = ["exitDetector"]

        self._setHistory()


        mes.currentMessage("initializing starting internal state")
        self.currentState = self.perceive(self.problemStateDefinition)        #PARAMETRIZE
        currentGState = self.perceive(self.goalStateDefinition)
        self.rsSize = 1 if not isinstance(currentGState,list) else len(currentGState)                             #PARAMETRIZE

        mes.settingMessage("Action-state values table")
        self.qAgent = qLA.batchBoltzmann(self,self.rsSize, len(self.currentState), self.environment.world.numActions, self.livePar.batchSize)
        mes.setMessage("Action-state values table")

        self.graphic = graphic

        if (self.graphic):
            mes.currentMessage("initializing render")
            self.environment._initGraph(self.goalStateDefinition)

    def act(self, rs):
        mes.currentMessage("selecting action according to current beleived state")
        action = self.qAgent.policy(self.currentState, rs)

        mes.currentMessage("performing action: " + str(action))
        (self.environment).performAction(action)  # here actual state is updated
        self.updatePerceivedTime()

        mes.currentMessage("perceiving")
        newState = self.perceive(self.problemStateDefinition)  # PARAMETRIZE
        mes.message("current problem state: " + str(newState))
        newGState = self.perceive(self.goalStateDefinition)
        reward = self.R(newGState)
        (self.sensoryHistory).append(newState+newGState)
        mes.currentMessage("Reward:" + str(reward))


        mes.currentMessage("observing transition")
        transition = ((self.currentState, action, newState), reward)
        (self.transitionHistory).append(transition)
        mes.currentMessage("learning from previous transition: ")
        self.qAgent.learn(transition)

        (self.stateHistory).append(self.currentState)

        mes.settingMessage("current beleived state from (" + str(self.currentState) + ")")
        self.currentState = newState
        mes.setMessage("current believed state to (" + str(self.currentState) + ")")

        #if (self.graphic):
            #(self.environment).changeBelief()

    def R(self, goalDetection):
        rs = []

        if isinstance(goalDetection, list):
            for i in goalDetection:
                if(i):
                    rs.append((self.livePar).goalReward)
                else:
                    rs.append((self.livePar).baseReward)
        else:
            if (goalDetection):
                rs.append((self.livePar).goalReward)
            else:
                rs.append((self.livePar).baseReward)

        return rs

    def nSteps(self, steps, rs):
        for i in range(steps): self.act(rs)

    def perceive(self, definition):
        #return (self.environment).currentPerception()
        percept = []

        for name in definition:
            sens = self.sensors[(self.sensorsNames).index(name)]
            percept += sens((self.environment).interrogateEnvironment, self)

        return list(map(int,percept))

    def mapInternalState(self, sensors):
        return self.environment.world._hashFun(sensors)

    '''
    def splitInternalState(self, state, definition):
        partition = []

        for part in definition:
            partition.append( 1 if state[(self.sensorsNames).index(part)] else 0)

        return partition[0] if len(partition)==1 else  partition
    '''

    def updatePerceivedTime(self):
        self.time += 1

    def _setHistory(self):
        mes.currentMessage("initializing perception history")
        self.sensoryHistory = []

        mes.currentMessage("initializing states history")
        self.stateHistory = []

        mes.currentMessage("initializing transition history")
        self.transitionHistory = []

        mes.currentMessage("initializing perceived time")
        self.time = 0

    def reset(self):
        self.currentState = (self.stateHistory)[0]
        self.environment._reset()
        self.qAgent.reset()

        self._setHistory()

    def __del__(self):
        self.currentState = self.sensoryHistory = self.transitionHistory = self.time = 0
        del self.qAgent
        print (self.__class__.__name__, "has been deleted")
