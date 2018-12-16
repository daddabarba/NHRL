import numpy as np

import vecStats as stats
import LSTM


class QL():

	def __init__(self, nStates, nActions, pars):

		self.pars = pars

		self.nStates = nStates
		self.nActions = nActions

		self._alpha = self.pars.learningRate
		self._gamma = self.pars.discountFactor

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


class TabularQL(QL):

	def __init__(self, nStates, nActions, pars):

		super(TabularQL, self).__init__(nStates, nActions, pars)

		self.table = np.zeros([nStates, nActions])

	def Q(self, s):
		return self.table[s]

	def update_Q(self, s, a, predicted):
		self.Q[s][a] = predicted


class NeuralQL(QL):

	def __init__(self, stateSize, nActions, pars):
		super(NeuralQL, self).__init__(stateSize, nActions, pars)

		self.net = LSTM.LSTM(stateSize, self.pars.rnnSize, nActions)

	def Q(self, s):
		return self.net(s)

	def update_Q(self, s, a, predicted):

		self.net.state_update()

		target = np.zeros(self.nActions)
		target[a] += predicted

		self.net.train(s, target)


class Boltzman(QL):

	def Pi(self, s):
		vals = np.exp(self.Q(s)/self.T())
		return vals/(vals.sum())

	def T(self):
		t = self.pars.startPoint - self.pars.speed * self.pars.time
		return ((np.e ** t) / ((np.e ** t) + 1)) * self.pars.height + self.pars.lowBound


class nStepQL(NeuralQL):

	def __init__(self, stateSize, nActions, pars):
		super(nStepQL, self).__init__(stateSize, nActions, pars)

		self._lambda = self.pars.batchSize

		self.S = np.zeros((self._lambda+1, 1, stateSize))
		self.states = np.zeros((self._lambda+1, self.pars.rnnSize))
		self.R = np.zeros(self._lambda+1)
		self.r_tot = 0
		self.A = np.zeros(self._lambda+1)

		self._gamma = self._gamma ** self._lambda

		self.cnt = 0

		self.factor = self._gamma ** (self._lambda-1)
		self.remove = (1 / self._gamma)

	def update(self, s1, a, r, s2):

		#Assume s2 will be s1 in next iteration

		if self.cnt < self._lambda:

			self.S[self.cnt][-1] += s1
			self.A[self.cnt] += a
			self.r_tot += (self._lambda**self.cnt)*r
			self.R[self.cnt] += r

			self.states[self.cnt] += self.net.state()
			self.net.state_update()

			self.cnt += 1

		else:

			self.S[-1][-1] += (s1 - self.S[-1][-1])

			self.states[self.cnt] += self.net.state()
			self.net.state_update()

			with LSTM.State_Set(self.net, self.states[0]):
				super(nStepQL, self).update(self.S[0:1], self.A[0], self.R, self.S)

			self.A[-1] = a
			self.r_tot = (self.r_tot - self.R[0])*self.remove + self.factor*r
			self.R[-1] = r

			self.S = np.roll(self.S, -1, axis=0)
			self.A = np.roll(self.A, -1, axis=0)
			self.R = np.roll(self.R, -1, axis=0)
			self.states = np.roll(self.states, -1, axis=0)


class hierarchy():

		def __init__(self, nStates, nActions, pars, QLCls, struc=[1]):

			self.pars = pars

			self.nStates = nStates
			self.nActions = nActions

			# Add primitve actions to structure
			struc += [nActions]

			# Initialize layer constructor
			vecCLS = np.vectorize(QLCls)
			rep = np.repeat

			# Build hierarchy of policies
			self.demons = np.array([vecCLS(rep(nStates, struc[i]), rep(struc[i+1], struc[i]), rep(pars, struc[i]))
										for i in range(len(struc)-1)], dtype=object)

			#Keep track of stats
			initStats = np.vectorize(stats.Stats)

			self.stats = np.array([initStats(rep(0.0, struc[i])) for i in range(self.demons.size)], dtype=object)
			self.layerStats = np.array([stats.Stats() for i in range(self.demons.size)], dtype=object)

			self.topDemonStats = stats.Stats()

			# Build empty state (pdist on actions) and Q (hierarchy of state-action utilities) arrays
			self.__initStateVariables(len(struc))

			# Vectorize QL methods
			self.layerPi = np.vectorize(QLCls.Pi, signature='(),(i)->(n)', otypes=[QLCls])
			self.layerUpdate = np.vectorize(QLCls.update, signature='(),(i),(),(),(i)->()', otypes=[QLCls])

			self.layerUpdateStats = np.vectorize(stats.Stats.update_stats, signature='(),(i),() -> ()', otypes=[stats.Stats])

			self.abstractLayer = np.vectorize(self.actionAbstraction, signature='(),(),()->()', otypes=[hierarchy])

		def __call__(self, state):
			return np.random.choice(self.nActions, p=self.Pi(state))

		def __initStateVariables(self, size):

			self.__beenTrained = True
			self.__likelihoods = np.empty(size - 1, dtype=object)

			self.PiVec = np.empty(size, dtype=object)
			self.PiVec[0] = np.array(1.0)

		def __getLikelihoods(self, state):

			for i in range(self.__likelihoods.size):
				self.__likelihoods[i] = self.layerPi(self.demons[i], state)

			return self.__likelihoods

		def Pi(self, state):

			self.__beenTrained = False
			self.__getLikelihoods(state)

			for i in range(1, self.PiVec.size):
				self.PiVec[i] = self.PiVec[i-1].dot(self.__likelihoods[i-1])

			return self.PiVec[-1]

		def update(self, s1, a, r, s2):

			if self.__beenTrained:
				self.Pi(s1)

			# Train each demon with the respective weighted reward
			for i in range(self.demons.size):
				self.layerUpdate(self.demons[i], s1, a, r*self.PiVec[i], s2)

			self.__beenTrained = True

			# Update stats of each demon
			for i in range(self.stats.size):
				self.layerUpdateStats(self.stats[i], s1, self.PiVec[i])
				self.layerStats[i].update_stats(s1)

			# Update top policy's stats
			self.topDemonStats.update_stats(self.PiVec[1])

		def actionAbstraction(self, layer, demon):

			if (self.stats[layer][demon].getVar()/self.layerStats[layer].getVa()) > self.pars.SDMax:

				#Abstract policy at layer layer (with index demon)








