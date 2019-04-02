import numpy as np
import torch

import vecStats as stats
import LSTM

import random as rand
import copy


# BASIC ABSTRACT CLASS

class QL():
    def __init__(self, nStates, nActions, pars):
        self.pars = pars

        self.nStates = nStates
        self.nActions = nActions

        self._alpha = self.pars.learningRate
        self._gamma = self.pars.discountFactor

    def __copy__(self):
        pass

    def __call__(self, s):

        pDist = self.Pi(s)

        if isinstance(pDist, torch.Tensor):
            pDist = pDist.detach().numpy()

        return np.random.choice(self.nActions, p=pDist)

    def Q(self, s):
        pass

    def Pi(self, s):
        pDist = torch.zeros(self.nActions)
        pDist[torch.argmax(self.Q(s))] = 1.0
        return pDist

    def U(self, s):
        return torch.dot(self.Pi(s), self.Q(s))

    def update(self, s1, a, r, s2):

        update = r + self._gamma * self.U(s2)

        self.update_Q(s1, a, update)

    def update_Q(self, s, a, update):
        pass

    def addAction(self, i=0):
        self.nActions += 1

    def biasAction(self, a):
        return None

    def getParent(self):
        pass


# Q FUNCTION REPRESENTATION

class TabularQL(QL):
    def __init__(self, nStates, nActions, pars, table=None):

        super(TabularQL, self).__init__(nStates, nActions, pars)

        if table is None:
            self.table = np.zeros([nStates, nActions])
        else:
            self.table = table

    def __copy__(self):
        return self.__class__(self.nStates, self.nActions, self.pars, copy.deepcopy(self.table))

    def Q(self, s):
        return self.table[s]

    def update_Q(self, s, a, update):
        self.table[s][a] = self._alpha*self.Q(s)[a] + (1-self._alpha)*update

    def addAction(self, i=0):
        super(TabularQL, self).addAction(i)
        self.table = np.hstack((self.Q, self.Q[:, i:(i + 1)]))

        tweaked = self.biasAction(self.Q(i))

        if tweaked is not None:
            self.table[i] = tweaked[0]
            self.table[-1] = tweaked[1]

    def getParent(self):

        parentTable = self.table.sum(1) / self.table.shape[1]

        tweaked = self.biasAction(parentTable)

        if tweaked is not None:
            parentTable = np.transpose(np.array([tweaked[0], tweaked[1]]))
        else:
            parentTable = np.transpose(np.repeat(np.array([parentTable]), 2, axis=0))

        return self.__class__(self.nStates, nBrothers, self.pars, parentTable)

class NeuralQL(QL):
    def __init__(self, stateSize, nActions, pars, net=None):
        super(NeuralQL, self).__init__(stateSize, nActions, pars)

        if net is None:
            self.net = LSTM.QL_LSTM(stateSize, self.pars.rnnSize, nActions, self._alpha)
        else:
            self.net = net

    def __copy__(self):
        return self.__class__(self.nStates, self.nActions, self.pars, copy.deepcopy(self.net))

    def Q(self, s):
        return self.net(s)

    def update_Q(self, s, a, update):
        with LSTM.State_Set(self.net, self.net.hcState()):
            self.net.train(s, a, update)
        self.net.state_update()

    def addAction(self, i=0):
        super(NeuralQL, self).addAction(i)
        self.net.duplicate_output(i)

        _, _b = self.net.getMlp()

        tweaked = self.biasAction(_b[i])

        if tweaked is not None:
            _b[i] = tweaked[0]
            _b[-1] = tweaked[-1]

    def getParent(self):

        _w, _b = self.net.getMlp()

        _w = _w.sum(0) / _w.shape[0]
        _b = _b.sum(0) / _b.shape[0]

        tweaked_b = self.biasAction(_b)

        _w = np.array([_w, _w])

        if tweaked_b is not None:
            _b = np.array([tweaked_b[0], tweaked_b[1]])
        else:
            _b = np.array([_b, _b])

        newNet = copy.copy(self.net)
        newNet.setMlp(_w, _b)

        return self.__class__(self.nStates, 2, self.pars, newNet)

    def abstractState(self):
        return self.net.state()


# EXPLORATION

class Boltzman(QL):

    def Pi(self, s):

        exponents = self.Q(s) / self.T()
        exponents = exponents - torch.max(exponents)

        vals = torch.exp(exponents)
        pDist = vals / (vals.sum())

        return pDist

    def T(self):
        t = self.pars.startPoint - self.pars.speed * self.pars.time
        return ((np.e ** t) / ((np.e ** t) + 1)) * self.pars.height + self.pars.lowBound

    def biasAction(self, a):
        # Reduce both of ln(2) with a tweak between -0.25 and 0.25 to differentiate them
        k = rand.random() * 0.5 + 0.25
        return a + np.log(k), a + np.log(1-k)


# EXPLOITATION

class nStepQL(NeuralQL):

    def __init__(self, stateSize, nActions, pars, net=None):
        super(nStepQL, self).__init__(stateSize, nActions, pars, net)

        self._lambda = self.pars.batchSize

        self.S = np.zeros((self._lambda + 1, 1, stateSize))
        #self.states = np.zeros(self._lambda + 1, dtype=object)
        self.R = np.zeros(self._lambda + 1)
        self.r_tot = 0
        self.A = np.zeros(self._lambda + 1, dtype=int)

        self.roll = (1.0 / self._gamma)
        self.factor = self._gamma ** (self._lambda - 1)
        self._gamma = self._gamma ** self._lambda

        self.cnt = 0

    def __copy__(self):

        ret = super(nStepQL, self).__copy__()
        ret.setHistory(copy.deepcopy(self.S), copy.deepcopy(self.R), self.r_tot,
                       copy.deepcopy(self.A))

        return ret

    def setHistory(self, S, states, R, r_tot, A):

        self.S = S
        #self.states = states
        self.R = R
        self.r_tot = r_tot
        self.A = A

    def update(self, s1, a, r, s2):

        # Assume s2 will be s1 in next iteration

        if self.cnt < self._lambda:

            self.S[self.cnt][-1] += s1
            self.A[self.cnt] += a
            self.r_tot += (self._gamma ** self.cnt) * r
            self.R[self.cnt] += r

            # self.states[self.cnt] = self.net.hcState()
            # self.net.state_update()

            self.cnt += 1

        else:

            self.S[-1][-1] += (s1 - self.S[-1][-1])

            # self.states[self.cnt] = self.net.hcState()
            # self.net.state_update()



            #with LSTM.State_Set(self.net, self.states[0]):
            super(nStepQL, self).update(s1, a, r, s2)

            self.A[-1] = a
            self.r_tot = (self.r_tot - self.R[0]) * self.roll + self.factor * r
            self.R[-1] = r

            self.S = np.roll(self.S, -1, axis=0)
            self.A = np.roll(self.A, -1, axis=0)
            self.R = np.roll(self.R, -1, axis=0)


            # self.states = np.roll(self.states, -1, axis=0)


# HIERARCHICAL

class hierarchy():
    def __init__(self, nStates, nActions, pars, QLCls, struc=[], max=None):

        self.pars = pars

        self.nStates = nStates
        self.nActions = nActions

        # Add primitve actions to structure
        struc += [nActions]
        struc = [1] + struc

        if max:
            self.max = [1] + max
        else:
            self.max = None

        # Initialize layer constructor
        vecCLS = lambda nStates, nActions, pars, len: [ QLCls(nStates, nActions, pars) for i in range(len)]
        rep = np.repeat

        # Build hierarchy of policies

        self.demons = np.empty(len(struc) - 1, dtype=object)
        for i in range(len(struc) - 1):
            self.demons[i] = np.array(vecCLS(nStates, struc[i + 1], pars, struc[i]))

        # Keep track of stats
        initStats = np.vectorize(stats.Stats)

        self.stats = np.empty(self.demons.size, dtype=object)
        for i in range(self.demons.size):
            self.stats[i] = initStats(rep(0.0, struc[i]))

        self.layerStats = stats.Stats()

        self.topDemonStats = stats.Stats()

        # Build empty state (pdist on actions) and Q (hierarchy of state-action utilities) arrays
        self.__initStateVariables(len(struc))

        # Vectorize QL methods
        self.layerPi = lambda layer, s : np.array([demon.Pi(s).detach().numpy() for demon in layer])
        self.layerUpdate = lambda layer, s1, p, a, r, s2 : [[layer[i].update(s1,k,r*p[k][a],s2) for k in range(len(p))] for i in range(len(layer))]

        self.layerUpdateStats = np.vectorize(stats.Stats.update_stats, signature='(),(i),()->()', otypes=[stats.Stats])

        self.abstractLayer = lambda layer, demons : [self.actionAbstraction(layer, i) for i in demons]
        self.layerAddAction = lambda layer, i : [demon.addAction(i) for demon in layer]

    def __call__(self, state):
        return np.random.choice(self.nActions, p=self.Pi(state)[0])

    def __initStateVariables(self, size):

        self.__beenTrained = True
        self.__lastState = None

        self.__likelihoods = np.empty(size - 1, dtype=np.ndarray)

        self.PiVec = np.empty(size, dtype=np.ndarray)
        self.PiVec[0] = np.array([[1.0]])

        self.PostVec = np.empty(size, dtype=np.ndarray)
        self.PostVec[-1] = np.eye(self.nActions)

    def __getLikelihoods(self, state):

        for i in range(self.__likelihoods.size):
            self.__likelihoods[i] = self.layerPi(self.demons[i], state)

        return self.__likelihoods

    def Pi(self, state):

        if not self.__beenTrained and self.__lastState == state:
            return self.PiVec[-1].astype(float) / self.PiVec[-1].sum()

        self.__beenTrained = False
        self.__lastState = state
        self.__getLikelihoods(state)

        for i in range(1, self.PiVec.size):
            self.PiVec[i] = np.dot(self.PiVec[i - 1], self.__likelihoods[i - 1])
            self.PostVec[self.PostVec.size -1 -i] = np.dot(self.__likelihoods[self.PostVec.size -1 -i], self.PostVec[self.PostVec.size -i] )

        return self.PiVec[-1].astype(float) / self.PiVec[-1].sum()

    def update(self, s1, a, r, s2):

        if self.__beenTrained:
            self.Pi(s1)

        # Train each demon with the respective weighted reward
        # for i in range(self.demons.size):
            # self.layerUpdate(self.demons[i], s1, self.PostVec[i], r, s2)

        for demon in self.demons[-1]:
            demon.update(s1, a, r, s2)

        for i in range(0, len(self.demons)-1):
            self.layerUpdate(self.demons[i], s1, self.PostVec[i+1], a, r, s2)

        self.__beenTrained = True

        # Update stats of each demon

        state = self.abstractState(s1)

        for i in range(self.stats.size):
            self.layerUpdateStats(self.stats[i], state, self.PiVec[i])
            self.layerStats.update_stats(state)

        # Update top policy's stats
        self.topDemonStats.update_stats(self.PiVec[1])

        # Action abstraction (if possible)
        for i in range(1, self.demons.size):
            self.abstractLayer(i, np.array(range(self.demons[i].size)))

    def abstractState(self, s):
        return np.array(s)

    def actionAbstraction(self, layer, demon, override=False):

        if override or (
                ( self.layerStats.getVar() > 0 and self.stats[layer][demon].getVar() / self.layerStats.getVar()) > self.pars.SDMax and self.stats[layer][
            demon].getN() > 1):

            # Abstract policy at layer layer (with index demon)

            if self.max and len(self.demons[layer]) >= self.max[layer]:
                return

            if layer == 0:
                return

            if layer==1:
                self.topDemonStats.data[stats.VAR] = np.resize(self.topDemonStats.data[stats.VAR], len(self.demons[1])+1)
                self.topDemonStats.data[stats.VAR][-1] = self.topDemonStats.data[stats.VAR][demon]

            # Copy the demon itself
            self.demons[layer] = np.append(self.demons[layer], copy.deepcopy(self.demons[layer][demon]))

            # Demons in higher levels must be aware of new demon at lower layer
            self.layerAddAction(self.demons[layer - 1], demon)

            self.stats[layer][demon].scale(2.0)
            self.stats[layer] = np.append(self.stats[layer], copy.deepcopy(self.stats[layer][demon]))

            if layer == 1:
                self.topDemonStats.reshape_mean()

    def taskAbstraction(self, override=False):

        if (self.topDemonStats.getVar() == 0) or (self.topDemonStats.getN() == 0):
            return

        if self.max and len(self.demons) >= len(self.max):
            return

        norm = np.linalg.norm(self.__likelihoods[0] - self.topDemonStats.getMu()) / self.topDemonStats.getVar()

        if override or ((norm > self.pars.BNBound) and (self.topDemonStats.getN() > 1)):
            # Construct new top demon
            parentDemon = self.demons[0][0].getParent()

            self.demons = np.append(np.empty(1), self.demons)
            self.demons[0] = np.array([parentDemon], dtype=object)

            # Copy previous top demon
            self.demons[1] = np.append(self.demons[1], np.array([copy.copy(self.demons[1][0])]))

            # Generate new stats

            self.topDemonStats = stats.Stats(mu=np.array([0.5, 0.5]))

            self.stats = np.append(np.empty(1), self.stats)
            self.stats[0] = np.array([copy.copy(self.stats[1][0])])

            self.stats[1][0].scale(2.0)
            self.stats[1] = np.append(self.stats[1], copy.copy(self.stats[1][0]))

            # Update stats and state variables
            self.__initStateVariables(self.demons.size + 1)

    def getLikelihoods(self):
        return self.__likelihoods


# MIXTURES

class deepSoftmax(NeuralQL, Boltzman):
    def __init__(self, stateSize, nActions, pars, net=None):
        super(deepSoftmax, self).__init__(stateSize, nActions, pars, net)


class deepNSoftmax(deepSoftmax, nStepQL):
    def __init__(self, stateSize, nActions, pars, net=None):
        super(deepNSoftmax, self).__init__(stateSize, nActions, pars, net)


# CONCRETE HIERARCHIES

class hDeepSoftmax(hierarchy):
    def __init__(self, nStates, nActions, pars, struc=[], max=None):
        super(hDeepSoftmax, self).__init__(nStates, nActions, pars, deepSoftmax, struc, max)

    def abstractState(self, s):
        return self.demons[0][0].abstractState()


class hDeepNSoftmax(hierarchy):
    def __init__(self, nStates, nActions, pars, struc=[], max=None):
        super(hDeepNSoftmax, self).__init__(nStates, nActions, pars, deepNSoftmax, struc, max)

    def abstractState(self, s):
        return self.demons[0][0].abstractState()
