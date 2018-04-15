import sys

sys.path.append('../../../../messages/')
sys.path.append('../learningAgent/ann/')
sys.path.append('../agent/learningAgent/ann/')

import numpy as np
import tensorflow as tf
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

        self.previous_state = None
        self.last_action = None

    def policy(self, state, rs, learning=False):
        (a, stateValue) = self.argMaxQ(state, rs)
        mes.currentMessage("evaluating state at: " + str(stateValue) + ", with best action: " + str(a))

        mes.currentMessage("acting rationally")

        if not learning:
            self.previous_state = state
            self.last_action = a

        return a

    def learn(self, newState, r):
        mes.currentMessage("retrieving parameters")

        s1 = self.previous_state
        a = self.last_action
        s2 = newState

        if type(r) == type(0.0):
            r = [r]*(len(self.Q))

        mes.currentMessage("learning from transition <" + str(s1) + " , " + str(a) + " , " + str(s2) + " , " + str(r) + ">")

        _alpha = self.agent.livePar.learningRate
        _lambda = self.agent.livePar.discountFactor

        for i in range(len(self.Q)):

            valueNext = self.stateValue(s2,i) #Q[i][s2][self.policy(s2, i)]

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

    def stateValue(self, state, rs):
        return self.stateActionValue(state,self.policy(state, rs, learning=True),rs)

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
    def __init__(self, agent, rs, stateSize, nActions, session=None):
        self.sess = session
        super(neuralQL, self).__init__(agent, rs, stateSize, nActions)

    def updateQ(self, state, action, update, rs):
        target = self.stateValues(state, rs)
        target[action] = update

        ((self.Q)[rs]).train_neural_network(state, target)

    def stateValues(self, state, rs):
        return (((self.Q)[rs]).getLastPrediction(input=state) )

    def _setQ(self, rs, stateSize, nActions):
        self.Q = []

        if not self.sess:
            self.sess = tf.Session()

        for i in range(rs):
            (self.Q).append(lstm.LSTM(stateSize, _defRnnSize, nActions, self.agent.livePar.neuralLearningRate, session=self.sess))



class batchQL(neuralQL):
    def __init__(self, agent, rs, stateSize, nActions, batchSize, session=None):
        super(batchQL, self).__init__(agent, rs, stateSize, nActions, session)

        self.batchSize = batchSize
        self.currentBatch = []

    def learn(self, newState, reward):
        self.currentBatch.append(((self.previous_state,self.last_action, newState), reward))

        if len(self.currentBatch)>=self.batchSize:
            for ((s1,a,s2),r) in self.currentBatch:
                self.previous_state = s1
                self.last_action = a
                super(batchQL, self).learn(s2,r)

            self.currentBatch = []

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
            a = rand.randint(0, 3)
        else:
            mes.currentMessage("acting rationally, with p: " + str(dice))
            a = super(simAnneal, self).policy(state, rs)

        if not learning:
            self.previous_state = state
            self.last_action = a

        return a

class boltzmann(simAnneal):
    def _val(self, t):
        t = self.agent.livePar.startPoint - self.agent.livePar.speed*t
        return ((np.e**t)/((np.e**t)+1))*self.agent.livePar.height + self.agent.livePar.lowBound

    def getPDist(self, state, rs):
        values = np.power(np.e,self.stateValues(state,rs)/self._val(self.agent.time))
        return values/(values.sum())

    def stateValue(self, state, rs):
        return (self.getPDist(state, rs)*self.stateValues(state,rs)).sum();

    def policy(self, state, rs, learning=False):
        probabilities = self.getPDist(state,rs)
        dice = rand.uniform(a=0.0, b=1.0)

        for i in range(len(probabilities)):
            if probabilities[i]==float('inf'):
                sys.exit("Infinity encountered at" + str(self.agent.time))

            if dice<=probabilities[i]:
                if not learning:
                    self.previous_state = state
                    self.last_action = i

                return i
            dice -= probabilities[i]

        a = rand.randint(0,len(probabilities)-1)

        if not learning:
            self.previous_state = state
            self.last_action = a

        return a

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


##############################
#EXPLORATION and EXPLOITATION#
##############################

class qLAIA(simAnneal, interestQLA):
    def __init__(self, agent, rs, r, c):
        super(qLAIA, self).__init__(agent, rs, r, c)

class neuralSimAnneal(simAnneal, neuralQL):
    def __init__(self, agent, rs, r, c, session=None):
        super(neuralSimAnneal, self).__init__(agent, rs, r, c, session)

class neuralBoltzmann(boltzmann, neuralQL):
    def __init__(self, agent, rs, r, c, session=None):
        super(neuralBoltzmann, self).__init__(agent, rs, r, c, session)

class batchBoltzmann(boltzmann, batchQL):
    def __init__(self, agent, rs, r, c, batchSize, session=None):
        super(batchBoltzmann, self).__init__(agent, rs, r, c, batchSize, session)


