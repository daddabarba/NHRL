import sys

sys.path.append('../../../../messages/')

import numpy as np
import random as rand

import LSTM as lstm

import messages as mes

_defRnnSize = 128

class qLA():
    def __init__(self, agent, rs, nStates, nActions):
        self.agent = agent

        self.nStates = nStates
        self.nActions = nActions

        self._setQ(rs, nStates, nActions)

    def policy(self, state, rs, learning=False):
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
            '''
            mes.currentMessage(
                "Updating state (" + str(s1) + ") action (" + str(a) + ") interest from: " + str((self.I)[i][s1][a]))
            mes.currentMessage("Updated to: " + str((self.I)[i][s1][a]))
            '''

            valueNext = self.stateActionValue(s2,self.policy(s2, i, learning=True),i) #Q[i][s2][self.policy(s2, i)]

            mes.currentMessage("computing new state action value")
            memory = (_alpha) * (self.stateActionValue(s1,a,i))  #((self.Q)[i][s1][a])
            learning = (1 - _alpha) * (r[i] + _lambda * (valueNext))

            mes.settingMessage("new state action value")
            #(self.Q)[i][s1][a] = memory + learning
            self.updateQ(s1,a, memory+learning, i)

            mes.setMessage("new state action value")

    def updateQ(self, state, action, update, rs):
        (self.Q)[rs][state][action] = update

    def argMaxQ(self, state, rs):
        max = np.argmax(self.stateValues(state,rs))

        return (max, self.stateActionValue(state,max, rs))

    '''
    def stateValues(self, state, rs):
        V = ((self.Q)[rs]) - np.outer(np.min(self.Q[rs], 1), np.ones(len(self.Q[rs][state])))
        V /= (np.outer(np.sum(V, 1), np.ones(len(V[state]))))

        return V[state]
    '''

    def stateActionValue(self, state, action, rs):
        return  (self.stateValues(state, rs))[action]

    def stateValues(self, state, rs):
        return Q[rs][state]

    def _setQ(self, rs, r, c):
        max = self.agent.livePar.startQMax
        min = self.agent.livePar.startQMin

        len = max - min

        self.Q = ((np.random.rand(rs, r, c)) * len) + min

    def reset(self):
        self._setQ(self.agent.rsSize, self.nStates,
                   self.nActions)
        #self.I = (np.zeros((len(self.Q), len(self.Q[0]), len(self.Q[0][0])))) + 1

    def __del__(self):
        self.Q = self.I = 0
        print(self.__class__.__name__, "has been deleted")



class basicQL(qLA):
    def __init__(self, agent, nStates, nActions):
        super(basicQL, self).__init__(agent,1,nStates,nActions)



class neuralQL(qLA):
    def __init__(self, agent, rs, stateSize, nActions):
        super(neuralQL, self).__init__(agent, rs, stateSize, nActions)

    def updateQ(self, state, action, update, rs):
        target = self.stateValues(state, rs)
        target[action] = update

        ((self.Q)[rs]).train_neural_network(state, target)

    def stateValues(self, state, rs):
        return (((self.Q)[rs]).getLastPrediction(input=state) )

    def _setQ(self, rs, stateSize, nActions):
        self.Q = []

        for i in range(rs):
            (self.Q).append(lstm.LSTM(stateSize, _defRnnSize, nActions, self.agent.livePar.neuralLearningRate))



#############
#EXPLORATION#
#############
class simAnneal(qLA):
    def _val(self, t):
        return np.e**(self.agent.livePar.scheduleA - (np.e**self.agent.livePar.scheduleB)*t)

    def _invVal(self, t):
        return 1 - self._val(t)

    def _schedule(self, t):
        val = self._val(t)

        return val if (val > self.agent.livePar.scheduleThresh) else 0

    def policy(self, state, rs, learning=False):
        p = self._schedule(self.agent.time)
        mes.currentMessage("Schedule: " + str(p))

        dice = float(rand.randint(0, 100)) / 100

        if (dice <= p and not learning):
            mes.currentMessage("acting randomly, with p: " + str(dice))
            return rand.randint(0, 3)

        mes.currentMessage("acting rationally, with p: " + str(dice))
        return super(simAnneal, self).policy(state, rs)

class boltzmann(simAnneal):
    def _val(self, t):
        #return np.e**(self.agent.livePar.scheduleA - (np.e**self.agent.livePar.scheduleB)*t) + self.agent.livePar.scheduleThresh
        return self.agent.livePar.scheduleThresh

    def _invVal(self, t):
        return 1 - self._val(t)

    def getPDist(self, state, rs):
        values = np.power(np.e,self.stateValues(state,rs)/self._val(self.agent.time))
        return values/(values.sum())

    def policy(self, state, rs, learning=False):
        probabilities = self.getPDist(state,rs)
        dice = rand.uniform(a=0.0, b=1.0)

        for i in range(len(probabilities)):
            if probabilities[i]==float('inf'):
                sys.exit("Infinity encountered at" + str(self.agent.time))

            if dice<=probabilities[i]:
                return i
            dice -= probabilities[i]

        return rand.randint(0,len(probabilities)-1)

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

    def reset(self):
        super(interestQLA, self).reset()
        # self.I = (np.zeros((len(self.Q), len(self.Q[0]), len(self.Q[0][0])))) + 1

class qLAIA(simAnneal, interestQLA):
    def __init__(self, agent, rs, r, c):
        super(qLAIA, self).__init__(agent, rs, r, c)

class neuralSimAnneal(simAnneal, neuralQL):
    def __init__(self, agent, rs, r, c):
        super(neuralSimAnneal, self).__init__(agent, rs, r, c)

class neuralBoltzmann(boltzmann, neuralQL):
    def __init__(self, agent, rs, r, c):
        super(neuralBoltzmann, self).__init__(agent, rs, r, c)


