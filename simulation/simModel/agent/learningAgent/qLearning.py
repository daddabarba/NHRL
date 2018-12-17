import numpy as np

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
		return np.random.choice(self.nActions, p=self.Pi(s))

	def Q(self, s):
		pass

	def Pi(self, s):
		return (self.Q[s] >= np.max(self.Q[s])) + 0

	def U(self, s):
		return np.dot(self.Pi(s), self.Q(s))

	def update(self, s1, a, r, s2):

		utility_approx = r + self._gamma*self.U(s2)

		memory = self._alpha * self.Q(s1)[a]
		update = (1-self._alpha) * utility_approx

		self.update_Q(s1, a, memory + update)

	def update_Q(self, s, a, predicted):
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

	def update_Q(self, s, a, predicted):
		self.Q[s][a] = predicted

	def addAction(self, i=0):
		super(TabularQL, self).addAction(i)
		self.Q = np.hstack((self.Q, self.Q[:, i:(i+1)]))

		tweaked = self.biasAction(self.Q[i])

		if tweaked is not None:
			self.Q[i] = tweaked[0]
			self.Q[-1] = tweaked[1]

	def getParent(self):

		parentTable = self.table.sum(1)/self.table.shape[1]

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
			self.net = LSTM.LSTM(stateSize, self.pars.rnnSize, nActions)
		else:
			self.net = net

	def __copy__(self):
		return self.__class__(self.nStates, self.nActions, self.pars, copy.deepcopy(self.net))

	def __call__(self, s):

		ret = super(NeuralQL, self).__call__(s)

		self.net.state_update()
		return ret

	def Q(self, s):
		return self.net(s)

	def update_Q(self, s, a, predicted):

		target = np.zeros(self.nActions)
		target[a] += predicted

		self.net.train(s, target)

	def addAction(self, i=0):
		super(NeuralQL, self).addAction(i)
		self.net.duplicate_output(i)

		_, _b = self.net.getMlp()

		tweaked = self.biasActions(_b[i])

		if tweaked is not None:
			_b[i] = tweaked[0]
			_b[-1] = tweaked[-1]

	def getParent(self):

		_w, _b = self.net.getMlp()

		_w = _w.sum(0)/_w.shape[0]
		_b = _b.sum(0)/_b.shape[0]

		tweaked_b = self.biasAction(_b)

		_w = np.array([_w, _w])

		if tweaked_b is not None:
			_b = np.array([tweaked_b[0], tweaked_b[1]])
		else:
			_b = np.array([_b, _b])

		newNet = copy.copy(self.net)
		newNet.setMlp(_w, _b)

		return self.__class__(self.nStates, nBrothers, self.pars, newNet)

	def abstractState(self):
		return self.net.state()


# EXPLORATION

class Boltzman(QL):

	def Pi(self, s):
		vals = np.exp(self.Q(s)/self.T())
		return vals/(vals.sum())

	def T(self):
		t = self.pars.startPoint - self.pars.speed * self.pars.time
		return ((np.e ** t) / ((np.e ** t) + 1)) * self.pars.height + self.pars.lowBound

	def biasAction(self, a):

		# Reduce both of ln(2) with a tweak between -0.25 and 0.25 to differentiate them
		k = rand.random()*0.5 + 0.25

		return a + np.log(1/k), a + np.log((k-1)/k)


# EXPLOITATION

class nStepQL(NeuralQL):

	def __init__(self, stateSize, nActions, pars, net=None):
		super(nStepQL, self).__init__(stateSize, nActions, pars, net)

		self._lambda = self.pars.batchSize

		self.S = np.zeros((self._lambda+1, 1, stateSize))
		self.states = np.zeros(self._lambda+1, dtype=object)
		self.R = np.zeros(self._lambda+1)
		self.r_tot = 0
		self.A = np.zeros(self._lambda+1, dtype=int)

		self._gamma = self._gamma ** self._lambda

		self.cnt = 0

		self.factor = self._gamma ** (self._lambda-1)
		self.remove = (1 / self._gamma)

	def __copy__(self):

		ret = super(nStepQL, self).__copy__()
		ret.setHistory(copy.deepcopy(self.S), copy.deepcopy(self.states), copy.deepcopy(self.R), r_tot, copy.deepcopy(self.A))

		return ret

	def setHistory(self, S, states, R, r_tot, A):

		self.S = S
		self.states = states
		self.R = R
		self.r_tot = r_tot
		self.A = A

	def update(self, s1, a, r, s2):

		#Assume s2 will be s1 in next iteration

		if self.cnt < self._lambda:

			self.S[self.cnt][-1] += s1
			self.A[self.cnt] += a
			self.r_tot += (self._lambda**self.cnt)*r
			self.R[self.cnt] += r

			self.states[self.cnt] = self.net.hcState()
			self.net.state_update()

			self.cnt += 1

		else:

			self.S[-1][-1] += (s1 - self.S[-1][-1])

			self.states[self.cnt] = self.net.hcState()
			self.net.state_update()

			with LSTM.State_Set(self.net, self.states[0]):
				super(nStepQL, self).update(self.S[0:1], self.A[0], self.r_tot, self.S)

			self.A[-1] = a
			self.r_tot = (self.r_tot - self.R[0])*self.remove + self.factor*r
			self.R[-1] = r

			self.S = np.roll(self.S, -1, axis=0)
			self.A = np.roll(self.A, -1, axis=0)
			self.R = np.roll(self.R, -1, axis=0)
			self.states = np.roll(self.states, -1, axis=0)

# HIERARCHICAL

class hierarchy():

		def __init__(self, nStates, nActions, pars, QLCls, struc=[]):

			self.pars = pars

			self.nStates = nStates
			self.nActions = nActions

			# Add primitve actions to structure
			struc += [nActions]
			struc = [1] + struc

			# Initialize layer constructor
			vecCLS = np.vectorize(QLCls)
			rep = np.repeat

			# Build hierarchy of policies
			self.demons = np.array([vecCLS(rep(nStates, struc[i]), rep(struc[i+1], struc[i]), rep(pars, struc[i]))
										for i in range(len(struc)-1)], dtype=object)

			#Keep track of stats
			initStats = np.vectorize(stats.Stats)

			self.stats = np.array([initStats(rep(0.0, struc[i])) for i in range(self.demons.size)], dtype=object)
			self.layerStats = stats.Stats()

			self.topDemonStats = stats.Stats()

			# Build empty state (pdist on actions) and Q (hierarchy of state-action utilities) arrays
			self.__initStateVariables(len(struc))

			# Vectorize QL methods
			self.layerPi = np.vectorize(QLCls.Pi, signature='(),(i)->(n)', otypes=[QLCls])
			self.layerUpdate = np.vectorize(QLCls.update, signature='(),(i),(),(),(i)->()', otypes=[QLCls])

			self.layerUpdateStats = np.vectorize(stats.Stats.update_stats, signature='(),(i),()->()', otypes=[stats.Stats])

			self.abstractLayer = np.vectorize(self.actionAbstraction, signature='(),(),()->()', otypes=[hierarchy])
			self.layerAddAction = np.vectorize(QLCls.addAction, signature='(),()->()', otypes=[QLCls])

		def __call__(self, state):
			return np.random.choice(self.nActions, p=self.Pi(state))

		def __initStateVariables(self, size):

			self.__beenTrained = True
			self.__likelihoods = np.empty(size - 1, dtype=np.ndarray)

			self.PiVec = np.empty(size, dtype=np.ndarray)
			self.PiVec[0] = np.array([1.0])

		def __getLikelihoods(self, state):

			for i in range(self.__likelihoods.size):
				self.__likelihoods[i] = self.layerPi(self.demons[i], state)

			return self.__likelihoods

		def Pi(self, state):

			self.__beenTrained = False
			self.__getLikelihoods(state)

			for i in range(1, self.PiVec.size):
				self.PiVec[i] = self.PiVec[i-1].dot(self.__likelihoods[i-1])

			return self.PiVec[-1].astype(float)/self.PiVec[-1].sum()

		def update(self, s1, a, r, s2):

			if self.__beenTrained:
				self.Pi(s1)

			# Train each demon with the respective weighted reward
			for i in range(self.demons.size):
				self.layerUpdate(self.demons[i], s1, a, r*self.PiVec[i], s2)

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
				self.abstractLayer(self, i, np.array(range(self.demons[i].size)))

		def abstractState(self, s):
			return np.array(s)

		def actionAbstraction(self, layer, demon):

			if (self.stats[layer][demon].getVar()/self.layerStats.getVa()) > self.pars.SDMax and self.stats[layer][demon].getN()>1:

				# Abstract policy at layer layer (with index demon)

				if layer == 0:
					return

				# Copy the demon itself
				self.demons[layer] = np.append(self.demons[layer], copy.copy(self.demons[layer][demon]))

				# Demons in higher levels must be aware of new demon at lower layer
				self.layerAddAction(self.demons[layer-1], demon)

				# Update stats and state variables

				self.__initStateVariables(self.demons.size + 1)

				self.stats[layer][demon].scale(2.0)
				self.stats[layer] = np.append(self.stats[layer], copy.copy(self.stats[layer][demon]))

				if layer == 1:
					self.topDemonStats.reshape_mean()

		def taskAbstraction(self):

			if (self.topDemonStats.getVar() == 0) or (self.topDemonStats.getN() == 0):
				return

			norm = np.linalg.norm(self.__likelihoods[0] - self.topDemonStats.getMu()) / self.topDemonStats.getVar()

			if (norm > self.pars.BNBound) and (self.topDemonStats.getN()>1):

				# Construct new top demon
				parentDemon = self.demons[0][0].getParent()

				self.demons = np.append(np.empty(1), self.demons)
				self.demons[0] = np.array([parentDemon])

				# Copy previous top demon
				self.demons[1] = np.append(self.demons[1], copy.copy(self.demons[1][0]))

				# Generate new stats

				self.topDemonStats = stats.Stats(mu=np.array([0.5,0.5]))

				self.stats = np.append(np.empty(1), self.stats)
				self.stats = np.array([copy.copy(self.stats[1][0])])

				self.stats[1][0].scale(2.0)
				self.stats[1] = np.append(self.stats[1], copy.copy(self.stats[1][0]))


# MIXTURES

class deepSoftmax(NeuralQL, Boltzman):

	def __init__(self, stateSize, nActions, pars, net=None):
		super(deepSoftmax, self).__init__(stateSize, nActions, pars, net)

class deepNSoftmax(deepSoftmax, nStepQL):

	def __init__(self, stateSize, nActions, pars, net=None):
		super(deepNSoftmax, self).__init__(stateSize, nActions, pars, net)


# CONCRETE HIERARCHIES

class hDeepSoftmax(hierarchy):

	def __init__(self, nStates, nActions, pars, struc=[]):
		super(hDeepSoftmax, self).__init__(nStates, nActions, pars, deepSoftmax, struc)

	def abstractState(self, s):
		return self.demons[0][0].abstractState()


class hDeepNSoftmax(hierarchy):

	def __init__(self, nStates, nActions, pars, struc=[]):
		super(hDeepNSoftmax, self).__init__(nStates, nActions, pars, deepNSoftmax, struc)

	def abstractState(self, s):
		return self.demons[0][0].abstractState()








