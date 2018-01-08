import sys

sys.path.append('../../../../messages/')

import numpy as np
import random as rand

import messages as mes


class qLA():
    def __init__(self, agent, rs, nStates, nActions):
        self.agent = agent
        self._setQ(rs, nStates, nActions)

    def policy(self, state, rs):
        (a, stateValue) = self.argMaxQ(state, rs)
        mes.currentMessage("evaluating state at: " + str(stateValue) + ", with best action: " + str(a))

        mes.currentMessage("acting rationally")
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
            mes.currentMessage("Updated to: " + str((self.I)[i][s1][a]))

            valueNext = self.Q[i][s2][self.policy(s2, i)]

            mes.currentMessage("computing new state action value")
            memory = (_alpha) * ((self.Q)[i][s1][a])
            learning = (1 - _alpha) * (r[i] + _lambda * (valueNext))

            mes.settingMessage("new state action value")
            (self.Q)[i][s1][a] = memory + learning

            mes.setMessage("new state action value")

    def argMaxQ(self, state, rs):
        max = np.argmax(self.stateValues(state,rs))

        return (max, self.stateActionValue(state,max, rs))

    def stateValues(self, state, rs):
        V = ((self.Q)[rs]) - np.outer(np.min(self.Q[rs], 1), np.ones(len(self.Q[rs][state])))
        V /= (np.outer(np.sum(V, 1), np.ones(len(V[state]))))

        return V[state]

    def stateActionValue(self, state, action, rs):
        return (self.Q)[rs][state][action]

    def _val(self, t):
        return (np.e) ** (
            ((-(np.e) ** (self.agent.livePar.scheduleB)) / (self.agent.livePar.scheduleA)) * t)

    def _invVal(self, t):
        return 1-self._val(t)

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
        print(self.__class__.__name__, "has been deleted")

class basicQL(qLA):
    def __init__(self, nStates, nActions):
        super(basicQL, self).__init__(1,nStates,nActions)

class simAnneal(qLA):
    def _schedule(self, t):
        val = self._val(t)

        return val if (val > self.agent.livePar.scheduleThresh) else 0

    def policy(self, state, rs):
        p = self._schedule(self.agent.time)
        mes.currentMessage("Schedule: " + str(p))

        dice = float(rand.randint(0, 100)) / 100

        if (dice <= p):
            mes.currentMessage("acting randomly, with p: " + str(dice))
            return rand.randint(0, 3)

        mes.currentMessage("acting rationally, with p: " + str(dice))
        return super(simAnneal, self).policy(state, rs)


class interestQLA(qLA):
    def __init__(self, agent, rs, r, c):
        super(interestQLA,self).__init__(agent,rs,r,c)

        mes.settingMessage("Action-state interest values table")
        self.I = (np.zeros((len(self.Q), len(self.Q[0]), len(self.Q[0][0])))) + 1
        mes.setMessage("Action-state interest values table")

    def learn(self, transition):
        super(interestQLA, self).learn(transition)
        self._updateInterestState(transition)

    def argMaxQ(self, state, rs):
        boundedInterestes = (self.I)[rs][state]

        reDimQ = super(interestQLA, self).stateValues(state,rs)

        V = reDimQ + boundedInterestes
        max = np.argmax(V)

        return (max, super(interestQLA,self).stateActionValue(state,max, rs))

    def _updateInterestState(self, transition):

        s1 = transition[0][0]
        a = transition[0][1]

        r = transition[1]
        for i in range(len(r)):
            if (r[i] < 0):
                self._updateInterest(i, s1, a)

    def _updateInterest(self, rs, state, action):
        (self.I)[rs][state][action] *= (np.e) ** (
        ((-(np.e) ** self.agent.livePar.interestB) * self._invVal(self.agent.time)) / (self.agent.livePar.interestA))


class qLAIA(simAnneal, interestQLA):
    def __init__(self, agent, rs, r, c):
        super(qLAIA, self).__init__(agent, rs, r, c)