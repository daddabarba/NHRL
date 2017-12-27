import sys

sys.path.append('../environment/')

sys.path.append('../../../messages/')

import numpy as np
import environment as env

import random as rand

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
        self._setQ(self.environment.world._sizeRewardSignal, self.environment.world.numStates,
                   self.environment.world.numActions)
        mes.setMessage("Action-state values table")

        mes.settingMessage("Action-state interest values table")
        self.I = (np.zeros((len(self.Q), len(self.Q[0]), len(self.Q[0][0])))) + 1
        mes.setMessage("Action-state interest values table")

        self.graphic = graphic

        if (self.graphic):
            mes.currentMessage("initializing render")
            self.environment._initGraph()

    def act(self, rs):
        mes.currentMessage("selecting action according to current beleived state")
        action = self.policy(self.currentState, rs)

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
        self.learn(transition)

        (self.stateHistory).append(self.currentState)

        mes.settingMessage("current beleived state from (" + str(self.currentState) + ")")
        self.currentState = newState
        mes.setMessage("current believed state to (" + str(self.currentState) + ")")

        if (self.graphic):
            (self.environment).changeBelief()

    def perceive(self):
        return (self.environment).currentPerception()

    def mapInternalState(self, sensors):
        return self.environment.world._hashFun(sensors)

    def argMaxQ(self, state, rs):

        boundedInterestes = (self.I)[rs]

        reDimQ = ((self.Q)[rs]) - np.outer(np.min(self.Q[rs], 1), np.ones(len(self.Q[rs][state])))
        reDimQ /= (np.outer(np.sum(reDimQ, 1), np.ones(len(reDimQ[state]))))

        V = reDimQ + boundedInterestes
        max = 0

        max = np.argmax(V[state])

        return (max, (self.Q)[rs][state][max])

    def _val(self, t):
        return ((np.e) ** (-(self.livePar.scheduleB))) * (
        (np.e) ** ((((-(np.e) ** (self.livePar.scheduleB)) / (self.livePar.scheduleA)) * t) + (self.livePar.scheduleB)))

    def _schedule(self, t):
        val = self._val(t)

        return val if (val > self.livePar.scheduleThresh) else 0

    def _updateInterest(self, rs, state, action):

        (self.I)[rs][state][action] *= ((np.e) ** (
        (-(np.e) ** (self.livePar.interestB)) / (self.livePar.interestA / (self.time ** self.livePar.interestC))))

    def policy(self, state, rs):
        p = self._schedule(self.time)
        mes.currentMessage("Schedule: " + str(p))

        dice = float(rand.randint(0, 100)) / 100

        (a, stateValue) = self.argMaxQ(state, rs)
        mes.currentMessage("evaluating state at: " + str(stateValue) + ", with best action: " + str(a))

        if (dice <= p):
            mes.currentMessage("acting randomly, with p: " + str(dice))
            return rand.randint(0, 3)

        mes.currentMessage("acting rationally, with p: " + str(dice))
        return a

    def updatePerceivedTime(self):
        self.time += 1

    def learn(self, transition):
        mes.currentMessage("retrieving parameters")

        s1 = transition[0][0]
        a = transition[0][1]
        s2 = transition[0][2]

        r = transition[1]

        _alpha = self.livePar.learningRate
        _lambda = self.livePar.discountFactor

        for i in range(len(self.Q)):
            mes.currentMessage(
                "Updating state (" + str(s1) + ") action (" + str(a) + ") interest from: " + str((self.I)[i][s1][a]))

            if (r[i] < 0):
                self._updateInterest(i, s1, a)

            mes.currentMessage("Updated to: " + str((self.I)[i][s1][a]))

            valueNext = self.Q[i][s2][self.policy(s2, i)]

            mes.currentMessage("computing new state action value")
            memory = (_alpha) * ((self.Q)[i][s1][a])
            learning = (1 - _alpha) * (r[i] + _lambda * (valueNext))

            mes.settingMessage("new state action value")
            (self.Q)[i][s1][a] = memory + learning

            mes.setMessage("new state action value")

    def reset(self):
        self.currentState = (self.stateHistory)[0]
        self.environment._reset()

        self._setQ(self.environment.world._sizeRewardSignal, self.environment.world.numStates,
                   self.environment.world.numActions)
        self.I = (np.zeros((len(self.Q), len(self.Q[0]), len(self.Q[0][0])))) + 1

        self._setHistory()

    def _setQ(self, rs, r, c):
        max = self.livePar.startQMax
        min = self.livePar.startQMin

        len = max - min

        self.Q = ((np.random.rand(rs, r, c)) * len) + min

    def _setHistory(self):
        mes.currentMessage("initializing perception history")
        self.sensoryHistory = []

        mes.currentMessage("initializing states history")
        self.stateHistory = []

        mes.currentMessage("initializing transition history")
        self.transitionHistory = []

        mes.currentMessage("initializing perceived time")
        self.time = 0

    def __del__(self):
        self.currentState = self.sensoryHistory = self.transitionHistory = self.time = 0
        print (self.__class__.__name__, "has been deleted")
