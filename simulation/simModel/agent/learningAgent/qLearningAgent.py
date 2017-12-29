import sys

sys.path.append('../../../../messages/')

import numpy as np
import random as rand

import messages as mes

class qLA():
    def __init__(self, agent, rs, r, c):
        self.agent = agent
        self._setQ(rs, r, c)

        mes.settingMessage("Action-state interest values table")
        self.I = (np.zeros((len(self.Q), len(self.Q[0]), len(self.Q[0][0])))) + 1
        mes.setMessage("Action-state interest values table")

    def argMaxQ(self, state, rs):
        boundedInterestes = (self.I)[rs]

        reDimQ = ((self.Q)[rs]) - np.outer(np.min(self.Q[rs], 1), np.ones(len(self.Q[rs][state])))
        reDimQ /= (np.outer(np.sum(reDimQ, 1), np.ones(len(reDimQ[state]))))

        V = reDimQ + boundedInterestes
        max = 0

        max = np.argmax(V[state])

        return (max, (self.Q)[rs][state][max])

    def _val(self, t):
        return (np.e) ** (
            ((-(np.e) ** (self.agent.livePar.scheduleB)) / (self.agent.livePar.scheduleA)) * t)

    def _invVal(self, t):
        return (np.e) ** (
            (((np.e) ** (self.agent.livePar.scheduleB)) / (self.agent.livePar.scheduleA)) * t) - 1

    def _schedule(self, t):
        val = self._val(t)

        return val if (val > self.agent.livePar.scheduleThresh) else 0
        #

    def _updateInterest(self, rs, state, action):
        (self.I)[rs][state][action] *= (np.e)**( ((-(np.e)**self.agent.livePar.interestB)*self._invVal(self.agent.time))/(self.agent.livePar.interestA) )

    def policy(self, state, rs):
        p = self._schedule(self.agent.time)
        mes.currentMessage("Schedule: " + str(p))

        dice = float(rand.randint(0, 100)) / 100

        (a, stateValue) = self.argMaxQ(state, rs)
        mes.currentMessage("evaluating state at: " + str(stateValue) + ", with best action: " + str(a))

        if (dice <= p):
            mes.currentMessage("acting randomly, with p: " + str(dice))
            return rand.randint(0, 3)

        mes.currentMessage("acting rationally, with p: " + str(dice))
        return a

    def learn(self, transition):
        mes.currentMessage("retrieving parameters")

        s1 = transition[0][0]
        a = transition[0][1]
        s2 = transition[0][2]

        r = transition[1]

        _alpha = self.agent.livePar.learningRate
        _lambda = self.agent.livePar.discountFactor

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

    def _setQ(self, rs, r, c):
        max = self.agent.livePar.startQMax
        min = self.agent.livePar.startQMin

        len = max - min

        self.Q = ((np.random.rand(rs, r, c)) * len) + min

    def reset(self):
        self._setQ(self.agent.environment.world._sizeRewardSignal, self.agent.environment.world.numStates,
                   self.agent.environment.world.numActions)
        self.I = (np.zeros((len(self.Q), len(self.Q[0]), len(self.Q[0][0])))) + 1

    def __del__(self):
        self.Q = self.I = 0
        print (self.__class__.__name__, "has been deleted")