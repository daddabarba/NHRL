import numpy as np

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


class hierarchy(QL):

		def __init__(self, nStates, nActions, pars, QLCls, struc=[1]):

			super(hierarchy, self).__init__(nStates, nActions, pars)

			# Add primitve actions to structure
			struc += [nActions]

			# Initialize layer constructor
			vecCLS = np.vectorize(QLCls)
			rep = np.repeat

			# Build hierarchy of policies
			self.demons = np.array([vecCLS(rep(nStates, struc[i]), rep(struc[i+1], struc[i]), rep(pars, struc[i]))
										for i in range(0, len(struc)-1)])

			# Build empty state (pdist on actions) and Q (hierarchy of state-action utilities) arrays
			self.__initStateVariables(len(struc))

			# self.QVec = np.empty(len(struc)-1, dtype=object)
			# self.UVec = np.empty(len(struc)-1, dtype=object)

			# Vectorize QL methods
			self.layerPi = np.vectorize(QLCls.Pi, signature='(),(i)->(i)')
			# self.layerQ = np.vectorize(QLCls.Q, signature='(),(i)->(i)')
			# self.layerU = np.vectorize(QLCls.U, signature='(),(i)->(i)')

		def __initStateVariables(self, size):

			self.__lastState = None
			self.__likelihoods = np.empty(size - 1, dtype=object)

			self.PiVec = np.empty(size, dtype=object)
			self.PiVec[0] = np.array(1.0)

		# def Q(self, state):

			# for i in range(self.QVec.size):
				# self.QVec[i] = self.layerQ(self.demons[i], state)

			# return self.QVec
		
		# def U(self, state):

			# for i in range(self.UVec.size):
				# self.UVec[i] = self.layerU(self.demons[i], state)

			# return self.UVec


		def __getLikelihoods(self, state):

			for i in range(self.__likelihoods.size):
				self.__likelihoods[i] = self.layerPi(self.demons[i], state)

			return self.__likelihoods

		def Pi(self, state):

			self.__lastState = state
			self.__getLikelihoods(state)

			for i in range(1, self.PiVec.size):
				self.PiVec[i] = self.PiVec[i-1].dot(self.__likelihoods[i-1])

			return self.PiVec[-1]

		def update(self, s1, a, r, s2):
			return None

		def update_Q(self, s, a, predicted):
			return None












