import numpy as np
import random as rand

import messages as mes


class model:
    def __init__(self, size, numActions, rewardSet, transitionSet, startingState=0):
        if (not (isinstance(size, tuple)) or not (isinstance(rewardSet, list)) or (
            not (isinstance(transitionSet, list)) and type(transitionSet).__module__ != np.__name__)):
            mes.errorMessage("cannot match size/reward setting/transition setting type")
            del self
            return

        mes.settingMessage("number of action(s)")
        self.numActions = numActions
        mes.setMessage("number of action(s)")

        mes.settingMessage("number of dimension(s)")
        self.dimensions = len(size)
        mes.setMessage("number of dimension(s)")

        mes.settingMessage("dimensions intervals (upper boundaries)")
        self.size = size
        mes.setMessage("dimensions intervals (upper boundaries)")

        mes.settingMessage("starting state")
        self._startingState = startingState
        mes.setMessage("starting state")

        self.resetModel()

        if (self.dimensions > 3 or self.dimensions < 1):
            mes.errorMessage("not supportable dimension(s). Should be between 1 and 3")
            del self
            return

        mes.settingMessage("number of states")
        self.numStates = 1

        for i in range(0, self.dimensions):
            self.numStates *= self.size[i]
        mes.setMessage("number of states")

        mes.settingMessage("reward table and transition probabilistic distribution")
        if (self._setR(rewardSet) == -1 or self._setT(transitionSet) == -1):
            mes.errorMessage("incorrect reward/transition setting")
            del self
            return
        mes.setMessage("reward table and transition probabilistic distribution")

    def _hashFun(self, location):
        mes.currentMessage("converting to state location " + str(location) + " in model")

        if (isinstance(location, int)):
            return location

        if (not (isinstance(location, tuple)) or self.dimensions != len(location)):
            mes.errorMessage("converting location in unmatching space")
            return -1

        if (self.dimensions == 2):
            return (self.size)[1] * location[0] + location[1]

        return (self.size)[0] * (self.size)[1] * location[2] + (self.size)[1] * location[0] + location[1]

    def _invHashFun(self, state):
        mes.currentMessage("locating state " + str(state) + " in model")

        if (isinstance(state, tuple)):
            return state

        if (state < 0 or state >= self.numStates):
            mes.errorMessage("locating in unmatching space")
            return -1

        if (self.dimensions == 2):
            return (int(state / ((self.size)[1])), state % ((self.size)[1]))

        return ((state % ((self.size)[0] * (self.size)[1])) % ((self.size)[1]),
                (state % ((self.size)[0] * (self.size)[1])) / ((self.size)[1]),
                state / ((self.size)[0] * (self.size)[1]))

    def _setR(self, rewardSetting):
        mes.currentMessage("initializing reward table")

        self._sizeRewardSignal = len(rewardSetting)
        self.R = np.zeros((self.numStates, self._sizeRewardSignal))

        if (rewardSetting == []):
            return

        for k in range(self._sizeRewardSignal):
            rewardSet = rewardSetting[k]
            x = 0

            if (not (isinstance(rewardSet, list))):
                mes.errorMessage("wrong signal setting")
                return -1

            mes.currentMessage("setting reward signal (" + str(k) + ")")

            if (isinstance(rewardSet[x], float)):
                mes.settingMessage("general reward value")

                z = np.transpose(self.R)
                z[k] += rewardSet[x]

                self.R = np.transpose(z)

                x += 1
                mes.setMessage("general reward value")

            mes.currentMessage("reading rewards partial settings")
            for i in range(x, len(rewardSet)):
                partSetting = rewardSet[i]
                mes.settingMessage("partial setting (" + str(i) + "): " + str(partSetting))

                if (not (isinstance(partSetting, tuple)) or not (isinstance(partSetting[1], float))):
                    mes.errorMessage(
                        "wrong reward setting format. Stopped at reward signal (" + str(k) + "), setting (" + str(
                            i) + "): " + str(partSetting))
                    return -1

                if (isinstance(partSetting[0], tuple)):
                    mes.currentMessage("locating referred state")
                    state = self._hashFun(partSetting[0])

                if (state == -1):
                    mes.errorMessage("unable to map state")
                    return -1

                (self.R)[state][k] = partSetting[1]
                mes.setMessage("partial reward setting")

        return 0

    def _setT(self, transitionSet):
        if (not (isinstance(transitionSet, list))):
            if (transitionSet.ndim <= 1):
                mes.errorMessage("wrong transition setting format")
                return -1

            mes.currentMessage("overwriting transition probability distribution")
            self.T = transitionSet
            return 0

        mes.currentMessage("initializing transition probability distribution")
        self.T = np.zeros((self.numStates, self.numActions, self.numStates))

        mes.currentMessage("reading transitions partial settings")
        for i in range(0, len(transitionSet)):
            partSetting = transitionSet[i]
            mes.settingMessage("partial setting (" + str(i) + "): " + str(partSetting))

            if (not (isinstance(partSetting, tuple)) or not (isinstance(partSetting[0], tuple)) or not (
            isinstance((partSetting[0])[1], int)) or len(partSetting[0]) != 3 or not (
            isinstance(partSetting[1], float)) or partSetting[1] < -1 or partSetting[1] > 1):
                mes.errorMessage("wrong transition setting format")
                return -1

            if (isinstance((partSetting[0])[0], tuple)):
                mes.currentMessage("locating referred starting state")
                state = self._hashFun((partSetting[0])[0])
            else:
                mes.currentMessage("locating referred ending state")
                state = ((partSetting[0])[0])

            mes.currentMessage("setting transition action")
            action = (partSetting[0])[1]

            if (isinstance((partSetting[0])[2], tuple)):
                transitionState = (self._hashFun((partSetting[0])[2]))
            else:
                transitionState = ((partSetting[0])[2])

            if (state == -1 or transitionState == -1):
                mes.errorMessage("unable to map state(s)")
                return -1

            if (partSetting[1] >= 0):
                (self.T)[state][action][transitionState] = partSetting[1]
            else:
                (self.T)[state][action][transitionState] -= partSetting[1]

            mes.setMessage("partial transition setting")

        warnings = 0

        mes.currentMessage("checking causality")
        for s in range(self.numStates):
            for a in range(self.numActions):
                sum = 0

                for s2 in range(self.numStates):
                    sum += (self.T)[s][a][s2]

                if (sum != 1):
                    mes.warningMessage(
                        "transition may not happen, transition p in (" + str(s) + ") with action (" + str(
                            a) + ") of: " + str(sum))
                    warnings += 1

        return warnings

    def transition(self, action):
        mes.currentMessage("generating random probability")
        p = float(rand.randint(0, 100)) / 100
        state = self.currentState

        mes.currentMessage("P: " + str(p))

        for s in range(self.numStates):

            transitionP = (self.T)[state][action][s]

            if (p <= transitionP and transitionP != 0):
                mes.currentMessage("transitioning with transition P: " + str(transitionP))
                self.currentState = s
                mes.currentMessage("updating time")
                self.time += 1
                return 0

            p -= transitionP

        mes.errorMessage("could not compute transition")
        return -1

    def _resetModel(self):

        mes.settingMessage("current state")
        self.currentState = self._startingState
        mes.setMessage("current state")

        mes.settingMessage("time")
        self.time = 0
        mes.setMessage("time")

    def __del__(self):
        self.size = self.dimensions = self.numStates = self.numActions = self.T = self.R = 0
        print (self.__class__.__name__, "has been deleted")
