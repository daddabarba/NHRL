import modWorld as modw

import parameters as par

import messages as mes


class basic(modw.model):
    def __init__(self, size, rewardSet, startingState=0):
        if (not (isinstance(size, tuple)) or not (isinstance(rewardSet, list))):
            mes.errorMessage("cannot match size/reward setting type")
            del self
            return

        mes.settingMessage("number of action(s)")
        self.numActions = 4
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

        self._resetModel()

        if (self.dimensions > par.maxDim or self.dimensions < par.minDim):
            mes.errorMessage("not supportable dimension(s). Should be between 1 and 3")
            del self
            return

        mes.settingMessage("number of states")
        self.numStates = 1

        for i in range(0, self.dimensions):
            self.numStates *= self.size[i]
        mes.setMessage("number of states")

        mes.settingMessage("basic transitions")
        transitionSet = self._predefTransSet()
        mes.settingMessage("basic transitions")

        mes.settingMessage("reward table and transition probabilistic distribution")
        if (self._setR(rewardSet) == -1 or self._setT(transitionSet) == -1):
            mes.errorMessage("incorrect reward/transition setting")
            del self
            return
        mes.setMessage("reward table and transition probabilistic distribution")

    def _predefTransSet(self):
        transitionSet = []

        for s in range(self.numStates):
            for a in range(self.numActions):
                currentState = self._invHashFun(s)
                nextState = self._applyAction(currentState, a)

                transitionSet.append(((s, a, nextState), 1.0))

        return transitionSet

    def _applyAction(self, currentState, a):
        hor = [1, 0, -1, 0]
        ver = [0, -1, 0, 1]

        nextState = list(currentState)

        if (self._vertCan(currentState, a)):
            nextState[0] += ver[a]
        if (self._horCan(currentState, a)):
            nextState[1] += hor[a]

        return tuple(nextState)

    def _vertCan(self, nextState, a):
        upBound = (nextState[0] > 0) or ((nextState[0] == 0) and a != par.u)
        lowBound = (nextState[0] < ((self.size)[0] - 1)) or ((nextState[0] == ((self.size)[0] - 1)) and a != par.d)

        return upBound and lowBound

    def _horCan(self, nextState, a):
        leftBound = (nextState[1] > 0) or ((nextState[1] == 0) and a != par.l)
        rightBound = (nextState[1] < ((self.size)[1] - 1)) or ((nextState[1] == ((self.size)[1] - 1)) and a != par.r)

        return leftBound and rightBound


def _isHorizontal(line):
    return ((line[0]) != (line[2]))


def _isVertical(line):
    return ((line[1]) != (line[3]))


class maze(basic):
    def __init__(self, size, lines, rewardSet, startingState=0):
        if (not (isinstance(size, tuple)) or not (isinstance(rewardSet, list))):
            mes.errorMessage("cannot match size/reward setting/transition setting type")
            del self
            return

        mes.settingMessage("number of action(s)")
        self.numActions = 4
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

        self._resetModel()

        if (self.dimensions > par.maxDim or self.dimensions < par.minDim):
            mes.errorMessage("not supportable dimension(s). Should be between 1 and 3")
            del self
            return

        mes.settingMessage("number of states")
        self.numStates = 1

        for i in range(0, self.dimensions):
            self.numStates *= self.size[i]
        mes.setMessage("number of states")

        mes.settingMessage("basic transitions")
        transitionSet = self._predefTransSet()
        mes.settingMessage("basic transitions")
        mes.settingMessage("maze transitions")
        transitionSet = self._generateT(lines, transitionSet)
        mes.settingMessage("maze transitions")

        mes.settingMessage("reward table and transition probabilistic distribution")
        if (self._setR(rewardSet) == -1 or self._setT(transitionSet) == -1):
            mes.errorMessage("incorrect reward/transition setting")
            del self
            return
        mes.setMessage("reward table and transition probabilistic distribution")

    def _generateT(self, lines, T):

        for i in range(len(lines)):
            line = lines[i]
            T = self._addWall(T, line)

        return T

    def _addWall(self, T, line):
        maxX = (self.size)[1]
        maxY = (self.size)[0]

        if (_isHorizontal(line)):

            rd = line[1]
            ru = rd - 1

            if (rd > 0 and rd < maxX):
                for i in range(line[0], line[2]):
                    T.append((((ru, i), par.d, (rd, i)), 0.0))
                    T.append((((ru, i), par.d, (ru, i)), 1.0))
                    T.append((((rd, i), par.u, (ru, i)), 0.0))
                    T.append((((rd, i), par.u, (rd, i)), 1.0))

        elif (_isVertical(line)):

            cr = line[0]
            cl = cr - 1

            if (cr > 0 and cr < maxY):
                for i in range(line[1], line[3]):
                    T.append((((i, cl), par.r, (i, cr)), 0.0))
                    T.append((((i, cl), par.r, (i, cl)), 1.0))
                    T.append((((i, cr), par.l, (i, cl)), 0.0))
                    T.append((((i, cr), par.l, (i, cr)), 1.0))


        else:
            mes.errorMessage("cannot recognise wall")

        return T


class stochastic(basic):
    def __init__(self, size, rewardSet, startingState=0):
        super().__init__(size, rewardSet, startingState)

    def _predefTransSet(self):
        transitionSet = []

        for s in range(self.numStates):
            for a in range(self.numActions):
                currentState = self._invHashFun(s)

                nextState = self._applyAction(currentState, a)
                dev1 = self._applyAction(currentState, (a - 1) % (self.numActions))
                dev2 = self._applyAction(currentState, (a + 1) % (self.numActions))

                transitionSet.append(((s, a, nextState), (par.stP)))
                transitionSet.append(((s, a, dev1), -(par.divP)))
                transitionSet.append(((s, a, dev2), -(par.divP)))

        return transitionSet


class stochasticMaze(maze, stochastic):
    def __init__(self, size, lines, rewardSet, startingState=0):
        super().__init__(size, lines, rewardSet, startingState)

    def _addWall(self, T, line):
        maxX = (self.size)[1]
        maxY = (self.size)[0]

        if (_isHorizontal(line)):

            rd = line[1]
            ru = rd - 1

            if (rd > 0 and rd < maxX):
                for i in range(line[0], line[2]):
                    T.append((((ru, i), par.d, (rd, i)), 0.0))
                    T.append((((ru, i), par.d, (ru, i)), -(par.stP)))

                    T.append((((ru, i), par.l, (rd, i)), 0.0))
                    T.append((((ru, i), par.l, (ru, i)), -(par.divP)))

                    T.append((((ru, i), par.r, (rd, i)), 0.0))
                    T.append((((ru, i), par.r, (ru, i)), -(par.divP)))

                    T.append((((rd, i), par.u, (ru, i)), 0.0))
                    T.append((((rd, i), par.u, (rd, i)), -(par.stP)))

                    T.append((((rd, i), par.l, (ru, i)), 0.0))
                    T.append((((rd, i), par.l, (rd, i)), -(par.divP)))

                    T.append((((rd, i), par.r, (ru, i)), 0.0))
                    T.append((((rd, i), par.r, (rd, i)), -(par.divP)))

        elif (_isVertical(line)):

            cr = line[0]
            cl = cr - 1

            if (cr > 0 and cr < maxY):
                for i in range(line[1], line[3]):
                    T.append((((i, cl), par.r, (i, cr)), 0.0))
                    T.append((((i, cl), par.r, (i, cl)), -(par.stP)))

                    T.append((((i, cl), par.u, (i, cr)), 0.0))
                    T.append((((i, cl), par.u, (i, cl)), -(par.divP)))

                    T.append((((i, cl), par.d, (i, cr)), 0.0))
                    T.append((((i, cl), par.d, (i, cl)), -(par.divP)))

                    T.append((((i, cr), par.l, (i, cl)), 0.0))
                    T.append((((i, cr), par.l, (i, cr)), -(par.stP)))

                    T.append((((i, cr), par.u, (i, cl)), 0.0))
                    T.append((((i, cr), par.u, (i, cr)), -(par.divP)))

                    T.append((((i, cr), par.d, (i, cl)), 0.0))
                    T.append((((i, cr), par.d, (i, cr)), -(par.divP)))

        else:
            mes.errorMessage("cannot recognise wall")

        return T
