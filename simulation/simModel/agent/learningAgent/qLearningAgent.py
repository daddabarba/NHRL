import sys

import numpy as np
import random as rand

import LSTM

import messages as mes

import vecStats as stats

import aux

_defRnnSize = 5

class qLA():
    def __init__(self, agent, rs, nStates, nActions):
        self.agent = agent

        self.nStates = nStates
        self.nActions = nActions

        self._setQ(rs, nStates, nActions)

        self.previous_state = None
        self.last_action = None

        self.last_policy = None if self.size()>1 else 0

    def policy(self, state, rs, learning=False):
        (a, stateValue) = self.argMaxQ(state, rs)
        mes.currentMessage("evaluating state at: " + str(stateValue) + ", with best action: " + str(a))

        mes.currentMessage("acting rationally")

        if not learning:
            self.previous_state = state
            self.last_action = a

            self.last_policy = rs

        return a

    def learn(self, newState, r):
        mes.currentMessage("retrieving parameters")

        s1 = self.previous_state
        a = self.last_action
        s2 = newState

        if type(r) == tuple:
            vec_r = [None for i in range(len(self.Q))]
            vec_r[r[0]] = r[1]
            r = vec_r
        elif type(r)!= list:
            r = [r] * (len(self.Q))

        mes.currentMessage("learning from transition <" + str(s1) + " , " + str(a) + " , " + str(s2) + " , " + str(r) + ">")

        _alpha = self.agent.livePar.learningRate
        _gamma = self.agent.livePar.discountFactor

        for i in range(len(self.Q)):
            if r[i]:
                valueNext = self.stateValue(s2,i) #Q[i][s2][self.policy(s2, i)]

                mes.currentMessage("computing new state action value")
                memory = (_alpha) * (self.stateActionValue(s1,a,i))  #((self.Q)[i][s1][a])
                learning = (1 - _alpha) * self.updateValue(r[i], _gamma, valueNext)

                mes.settingMessage("new state action value")
                #(self.Q)[i][s1][a] = memory + learning
                self.updateQ(s1,a, memory+learning, i)

                mes.setMessage("new state action value")

    def updateValue(self, observations, _gamma, prediction):
        return observations + _gamma*prediction

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

    def size(self):
        return len(self.Q)

    def reset(self):
        self._setQ(self.agent.rsSize, self.nStates, self.nActions)

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

        ((self.Q)[rs]).train(state, target)

    def stateValues(self, state, rs):
        return self.Q[rs](state)

    def _setQ(self, rs, stateSize, nActions):
        self.Q = []

        for i in range(rs):
            (self.Q).append(LSTM.LSTM(stateSize, _defRnnSize, nActions, self.agent.livePar.neuralLearningRate))

    def copyAction(self, ind):
        for i in range(len(self.Q)):
            self.Q[i] = self.Q[i].duplicate_output(ind)

    def copyPolicy(self, ind):
        self.Q.append(LSTM.LSTM.copy_net(self.Q[ind]))

    def getNNState(self, rs):
        return self.Q[rs].state()

    def rec(self, rs):
        self.Q[rs].state_update()

class batchQL(neuralQL):
    def __init__(self, agent, rs, stateSize, nActions, batchSize, session=None):
        super(batchQL, self).__init__(agent, rs, stateSize, nActions, session)

        self.batchSize = batchSize
        self.currentBatch = []

    def learn(self, newState, reward):
        self.currentBatch.append(((self.previous_state,self.last_action, newState), reward))

        if len(self.currentBatch)>=self.batchSize:
            for ((s1,a,s2),r) in self.currentBatch:
                with aux.tempTransition(self, s1, a):
                    super(batchQL, self).learn(s2,r)

            self.currentBatch = []


class nstepQL(neuralQL):
    class observation:
        def __init__(self, gamma, _lambda):

            self.R = []
            self.P = []

            self.gamma = gamma
            self._lambda = _lambda

            self.tot = 0

            self.start = 0

            self.factor = self.gamma ** self._lambda
            self.remove = (1 / self.gamma)

        def update(self, val, action, state):

            if not state or (not action and action != 0):
                return

            self.R.append(val)
            self.P.append((state, action))

            if self.isExceeding():
                self.tot -= self.R[self.start]

                self.start += 1

                self.tot *= self.remove
                self.tot += val * self.factor

                if len(self.R) > (10 * self._lambda):
                    self.R = self.R[::-1][0:self._lambda][::-1]
                    self.P = self.P[::-1][0:self._lambda][::-1]

                    self.start = 0

            else:
                self.tot += val * (self.gamma ** (len(self.R) - 1))

        def getVal(self):
            return self.tot

        def getState(self):
            return self.P[self.start][0]

        def getAction(self):
            return self.P[self.start][1]

        def isFull(self):
            return len(self.R) >= (self._lambda + 1)

        def isExceeding(self):
            return len(self.R) > (self._lambda + 1)

    def __init__(self, agent, rs, stateSize, nActions, _lambda, session=None):
        super(nstepQL, self).__init__(agent, rs, stateSize, nActions, session)

        self.gamma = agent.livePar.discountFactor

        self._lambda = _lambda
        self.observations = [self.observation(self.gamma, _lambda) for i in range(len(self.Q))]

    def learn(self, newState, r):

        if type(r) == type(0.0):
            r = [r] * (len(self.Q))

        if type(r) == tuple:
            vec_r = [None for i in range(len(self.Q))]
            vec_r[r[0]] = r[1]
            r = vec_r

        for rs in range(len(r)):
            if r[rs]:
                self.observations[rs].update(r[rs], self.last_action, self.previous_state)

                if self.observations[rs].isFull():
                    with aux.tempTransition(self, self.observations[rs].getState(), self.observations[rs].getAction(), rs):
                        super(nstepQL, self).learn(newState, (rs,self.observations[rs].getVal()))


    def updateValue(self, observations, _gamma, prediction):
        return observations + (_gamma**(self._lambda+1)) * prediction

    def copyPolicy(self, ind):
        super(nstepQL, self).copyPolicy(ind)

        self.observations.append(self.observation(self.gamma, self._lambda))


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

            self.last_policy = rs

        return a

class boltzmann(simAnneal):
    def _val(self, time):
        t = self.agent.livePar.startPoint - self.agent.livePar.speed*time
        return ((np.e**t)/((np.e**t)+1))*self.agent.livePar.height + self.agent.livePar.lowBound

    def getPDist(self, state, rs):
        values = np.power(np.e,self.stateValues(state,rs)/self._val(self.agent.time))
        return values/(values.sum())

    def getPMat(self, state):
        M = []

        for rs in range(len(self.Q)):
            M.append(self.getPDist(state, rs))

        return np.array(M)

    def stateValue(self, state, rs):
        return (self.getPDist(state, rs)*self.stateValues(state,rs)).sum()

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

                    self.last_policy = rs

                return i
            dice -= probabilities[i]

        a = rand.randint(0,len(probabilities)-1)

        if not learning:
            self.previous_state = state
            self.last_action = a

            self.last_policy = rs

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

    def policy(self, state, rs, learning=False):
        ret = super(batchBoltzmann, self).policy(state, rs, learning)

        self.rec(rs)
        return ret

class nStepBoltzmann(boltzmann, temporalDifference):
    def __init__(self, agent, rs, r, c, _lambda, session=None):
        super(nstepQL, self).__init__(agent, rs, r, c, _lambda, session)

    def policy(self, state, rs, learning=False):
        ret = super(nstepQL, self).policy(state, rs, learning)

        self.rec(rs)
        return ret


#####################################
#HIERARCHICAL REINFORCEMENT LEARNING#
#####################################

######################
#ABSTRACT HIERARCHIES#
######################

class hierarchy():

    def __init__(self, agent, policyClass, stateSize, batchSize, nActions=None, structure=[1], max=None):
        self.agent = agent
        self.batch_size = batchSize
        self.stateSize = stateSize

        self.policyClass = policyClass

        self.max = max

        if nActions:
            structure = [nActions] + structure

        self.hierarchy = []
        for i in range(1,len(structure)):
            layer = policyClass(agent, structure[i], stateSize, structure[i-1], batchSize)
            self.hierarchy.append(layer)

        report_template = {'sd': 0.0, 'mu': np.zeros(_defRnnSize), 'N': 0}
        self.policy_data = [[report_template.copy() for i in range(layer_size)] for layer_size in structure[1:]]
        self.layer_data = [report_template.copy() for layer_size in structure[1:]]

        self.bottleneck_data = self.make_bottleneck_data(structure[-2])

    def make_bottleneck_data(self, size):
        return {'sd': 0.0, 'mu': np.zeros(size), 'N': 0}

    def policy(self, state, rs=0, layer=None):
        if not layer and layer!=0:
            layer = len(self.hierarchy) - 1

        action = self.hierarchy[layer].policy(state, rs)
        self.hierarchy[layer].rec(rs)

        if layer == len(self.hierarchy) - 1:
            self.updateBNData(state, rs)

        abstract_state = self.state_abstraction(state, layer, rs)
        self.updateData(abstract_state, rs, layer)

        if layer!=0:
            mes.currentMessage("action: " + str(action) + "/" + str(len(self.hierarchy[layer - 1].Q)-1))
            return self.policy(state, action, layer-1)

        return action

    def state_abstraction(self, state, layer, rs):
        return self.hierarchy[layer].getNNState(rs)

    def task_abstraction(self, rs=0):

        if self.max and len(self.max)<=len(self.hierarchy):
            mes.currentMessage("Reached maximum size for bottom-up abstraction")
            return

        mes.warningMessage("Getting network's parameters")

        ANN = self.hierarchy[-1].Q[rs]
        parameters = ANN.getCopy()

        mes.warningMessage("unrolling parameters")

        W = parameters['out'][lstm.out_weights_key]
        b = parameters['out'][lstm.out_bias_key]
        rnn = parameters['rnn']

        mes.warningMessage("Getting network shape")

        size = (np.shape(W)[1])

        mes.warningMessage("Computing new weights")

        new_w = (W * (1 / size)).sum(1)
        new_W = np.transpose(np.array([new_w, new_w]))

        mes.warningMessage("Computing new biases")

        _b = (b*size).sum()
        new_b = np.array([_b, _b])

        mes.warningMessage("Rolling weights and biases")

        W_pars = {lstm.out_weights_key: new_W, lstm.out_bias_key: new_b}

        mes.warningMessage("Copying policy")

        (self.hierarchy)[-1].copyPolicy(rs)
        #new_ANN = ANN.restart(ANN.input_size, rnn, W_pars, ANN.alpha, self.sess, ANN.scope)

        mes.warningMessage("Restaring top policy")

        self.hierarchy.append(self.policyClass(self.agent, 1, self.stateSize, 2, self.batch_size, None))
        self.hierarchy[-1].Q[0] = self.hierarchy[-1].Q[0].restart(ANN.input_size, rnn, W_pars, ANN.alpha, None, ANN.scope)

        mes.currentMessage("Adjusting stats")

        self.bottleneck_data = self.make_bottleneck_data(2)

        self.policy_data.append([self.policy_data[-1][rs].copy()])
        self.layer_data.append(self.layer_data[-1].copy())

        self.policy_data[-2][rs]['sd'] *= 1.0 / 4.0
        self.policy_data[-2][rs]['mu'] *= 1.0 / 2.0

        self.policy_data[-2].append(self.policy_data[-2][rs].copy())

        self.bottleneck_data['mu'] = np.array([0.5, 0.5])

    def action_abstraction(self, policy, layer):
        mes.currentMessage("Abstracting action")
        mes.currentMessage("Policy (" + str(layer) + "," + str(policy) + ") not specialized, splitting in subtasks")

        if self.max and (len(self.max)<=layer or self.max[layer]<=len(self.hierarchy[layer].Q)) :
            mes.currentMessage("Reached maximum size for layer %i" % layer)
            return

        if layer == len(self.hierarchy) - 1:
            return

        self.hierarchy[layer + 1].copyAction(policy)
        self.hierarchy[layer].copyPolicy(policy)

        self.policy_data[layer][policy]['sd'] *= 1.0/4.0
        self.policy_data[layer][policy]['mu'] *= 1.0/2.0
        self.policy_data[layer][policy]['N'] *= 1.0/2.0

        self.policy_data[layer].append(self.policy_data[layer][policy].copy())

        if layer == len(self.hierarchy)-2:
            self.bottleneck_data['mu'] = stats.reshape_mean(self.bottleneck_data['mu'])
            #self.make_bottleneck_data(len(self.hierarchy[-2].Q))

    def updateData(self, newState, policy, layer):
        self.policy_data[layer][policy] = stats.update_stats(self.policy_data[layer][policy], newState)
        self.layer_data[layer] = stats.update_stats(self.layer_data[layer], newState)

        WSS =  self.policy_data[layer][policy]['sd']
        TSS =  self.layer_data[layer]['sd']

        if(WSS/TSS) > self.agent.livePar.SDMax:
            self.action_abstraction(policy, layer)

    def updateBNData(self, state, rs):
        pDist = np.array(self.hierarchy[-1].getPDist(state, rs))

        if self.bottleneck_data['sd']==0 or self.bottleneck_data['N'] == 0:
            self.bottleneck_data = stats.update_stats(self.bottleneck_data, pDist)
            return

        norm = np.linalg.norm(pDist - self.bottleneck_data['mu'])/self.bottleneck_data['sd']

        if norm>self.agent.livePar.BNBound:
            self.task_abstraction()

        else:

            self.bottleneck_data = stats.update_stats(self.bottleneck_data, pDist)

    def learn(self, newState, r):

        mes.currentMessage("Broadcasting reward to previous policy firing chain")

        if type(r)==list:
            r = r[0]

        for layer in self.hierarchy:

            mes.currentMessage("Current layer: %d"%self.hierarchy.index(layer))

            if layer.last_policy or layer.last_policy == 0:
                mes.currentMessage("Reward sent")

                reward = (layer.last_policy, r)
                layer.learn(newState, reward)

    def reset(self):
        for layer in self.hierarchy:
            layer.reset()


    def printHierarchy(self):
        desc = ""
        for layer in self.hierarchy:
            desc = desc + str(len(layer.Q)) + " , "
        return desc


class weightedHierarchy(hierarchy):

    def getMixture(self, state):
        mixture = [np.array([[1.0]])]

        for layer in list(range(1, len(self.hierarchy)))[::-1]:
            M = self.hierarchy[layer].getPMat(state)
            mixture.append(np.dot(mixture[-1], M))

        return mixture[::-1]

    def learn(self, newState, r):
        mes.currentMessage("Broadcasting reward to previous policy firing chain")

        if type(r) == list:
            r = r[0]

        mixture = self.getMixture(newState)

        for layer in self.hierarchy:

            mes.currentMessage("Current layer: %d" % self.hierarchy.index(layer))

            if layer.last_policy or layer.last_policy == 0:
                mes.currentMessage("Reward sent")

                reward = (r*mixture[self.hierarchy.index(layer)])[0]
                layer.learn(newState, reward)

######################
#CONCRETE HIERARCHIES#
######################

class hBatchBoltzmann(hierarchy):
    def __init__(self, agent, stateSize, batchSize, nActions=None, structure=[1]):
        super(hBatchBoltzmann, self).__init__(agent, batchBoltzmann, stateSize, batchSize, nActions, structure)

class hNStepsBoltzmann(hierarchy):
    def __init__(self, agent, stateSize, batchSize, nActions=None, structure=[1]):
        super(hTDBoltzmann, self).__init__(agent, nStepBoltzmann, stateSize, batchSize, nActions, structure)

class hTDWeightBoltzmann(weightedHierarchy):
    def __init__(self, agent, stateSize, batchSize, nActions=None, structure=[1]):
        super(hTDWeightBoltzmann, self).__init__(agent, tdBoltzmann, stateSize, batchSize, nActions, structure)