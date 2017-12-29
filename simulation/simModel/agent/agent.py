import sys

sys.path.append('../environment/')
sys.path.append('../agent/learningAgent/')

sys.path.append('../../../messages/')

import environment as env
import qLearningAgent as qLA

import agentPar as par

import messages as mes


class agent:
    def __init__(self, startingState="c", environment="../../files/maze.txt", graphic=1):

        mes.currentMessage("environment")
        self.environment = env.environment(environment, self, startingState, graphic)

        mes.settingMessage("live parameters")
        self.livePar = par.agentPar()
        mes.setMessage("live parameters")

        mes.currentMessage("initializing starting internal state")
        self.currentState = self.mapInternalState((self.perceive())[0])

        self._setHistory()

        mes.settingMessage("Action-state values table")
        self.qAgent = qLA.qLA(self,self.environment.world._sizeRewardSignal, self.environment.world.numStates,
                   self.environment.world.numActions)
        mes.setMessage("Action-state values table")

        self.graphic = graphic

        if (self.graphic):
            mes.currentMessage("initializing render")
            self.environment._initGraph()

    def act(self, rs):
        mes.currentMessage("selecting action according to current beleived state")
        action = self.qAgent.policy(self.currentState, rs)

        mes.currentMessage("performing action: " + str(action))
        (self.environment).performAction(action)  # here actual state is updated
        self.updatePerceivedTime()

        mes.currentMessage("perceiving")
        (sensors, reward) = self.perceive()  # dependent on current actual state
        (self.sensoryHistory).append(sensors)
        mes.currentMessage("computing current beleived state from SENSOR " + str(sensors))
        newState = self.mapInternalState(sensors)

        mes.currentMessage("observing transition")
        transition = ((self.currentState, action, newState), reward)
        (self.transitionHistory).append(transition)
        mes.currentMessage("learning from previous transition: ")
        self.qAgent.learn(transition)

        (self.stateHistory).append(self.currentState)

        mes.settingMessage("current beleived state from (" + str(self.currentState) + ")")
        self.currentState = newState
        mes.setMessage("current believed state to (" + str(self.currentState) + ")")

        if (self.graphic):
            (self.environment).changeBelief()

    def nSteps(self, steps, rs):
        for i in range(steps): self.act(rs)

    def perceive(self):
        return (self.environment).currentPerception()

    def mapInternalState(self, sensors):
        return self.environment.world._hashFun(sensors)

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
        print (self.__class__.__name__, "has been deleted")
