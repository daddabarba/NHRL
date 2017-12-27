import sys

sys.path.append('../agent/')
sys.path.append('../environment/metaModel/model/')
sys.path.append('../environment/metaModel/')

sys.path.append('../')

sys.path.append('../../GUI/')
sys.path.append('../../../messages/')

import numpy as np
import random as rand

import agent
import modWorld as modw
import preDefModels as pdm

import parameters as par

import graphic
import messages as mes


def _isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _convertFile(file):
    maze = open(file, "r+")

    maze.read(9)
    lines = []

    maxX = -1
    maxY = -1

    while (True):
        line = []

        for i in range(0, 4):
            maze.read(5)

            p = int(maze.read(1))
            c = maze.read(1)

            while (_isNumber(c)):
                p = p * 10 + int(c)
                c = maze.read(1)

            p = int((p - 2) / 16)
            line.append(p)

            if (not (i % 2) and p > maxX):
                maxX = p
            if ((i % 2) and p > maxY):
                maxY = p

        lines.append(line)
        c = maze.read(4)
        c = maze.read(1)

        if (c == "."):
            break
        maze.read(8)

    maze.close()

    return (lines, (maxY, maxX))


def _isHorizontal(line):
    return line[0] != line[2]


def _isVertical(line):
    return line[1] != line[3]


def _convertLines(lines, size):
    maxX = size[1]
    maxY = size[0]

    maps = np.zeros((maxY, maxX, 4))

    for i in range(len(lines)):
        line = lines[i]
        maps = _addLine(maps, line, size)

    return maps


def _addLine(maps, line, size):
    maxX = size[1]
    maxY = size[0]

    if (_isHorizontal(line)):

        rd = line[1]
        ru = rd - 1

        for i in range(line[0], line[2]):
            if (rd < maxX):
                maps[rd][i][1] = 1
            if (ru >= 0):
                maps[ru][i][3] = 1

    elif (_isVertical(line)):

        cr = line[0]
        cl = cr - 1

        for i in range(line[1], line[3]):
            if (cr < maxY):
                maps[i][cr][2] = 1
            if (cl >= 0):
                maps[i][cl][0] = 1

    else:
        mes.errorMessage("cannot recognise wall")

    return maps


def _getInterestPoints(maps, size):
    goals, impasses, crossRoads, food, water = [], [], [], [], []

    for i in range(0, size[0]):
        for k in range(0, size[1]):
            isUpperExit = i == 0 and maps[i][k][1] == 0
            isLowerExit = i == (size[0] - 1) and maps[i][k][3] == 0
            isRightExit = k == (size[1] - 1) and maps[i][k][0] == 0
            isLeftExit = k == 0 and maps[i][k][2] == 0

            isExit = isUpperExit or isLowerExit or isRightExit or isLeftExit

            if (isExit):
                goals.append((i, k))

            else:
                isBorder = (i <= int(size[0] / 5) or i >= int(4 * (size[0] / 5))) or (
                k < int(size[1] / 5) or k > int(4 * (size[1] / 5)))
                isCentral = (i >= int(2 * (size[0] / 4)) and i <= int(3 * (size[0] / 4))) or (
                k >= int(2 * (size[1] / 4)) and k <= int(3 * (size[1] / 4)))

                walls = 0
                for j in range(0, 4):
                    walls += maps[i][k][j]

                if (walls >= 3):
                    impasses.append((i, k))

                elif (walls <= 1):
                    crossRoads.append((i, k))

                else:
                    p = float(rand.randint(0, 100)) / 100

                    if (isBorder and p < par.waterP):
                        water.append((i, k))
                    elif (isCentral and p < par.foodP):
                        food.append((i, k))

    return (goals, impasses, crossRoads, food, water)


def _preDefRewardSet(interestPoints):
    rewardSet = []

    for k in range(len(interestPoints)):

        signalSet = [(par.baseReward)]

        for i in range(len(interestPoints[k])):
            signalSet.append((interestPoints[k][i], par.signalReward))

        rewardSet.append(signalSet)

    return rewardSet


class environment:
    def __init__(self, fileName, agent, startingState, graph):
        mes.currentMessage("linking agent")
        (self.agent) = agent
        mes.currentMessage("converting SVG file into lines format")
        (self.lines, self.size) = _convertFile(fileName)
        mes.currentMessage("converting lines format into map format")
        (self.maps) = _convertLines(self.lines, self.size)

        mes.currentMessage("retreiving interest points")
        self.interestPoints = _getInterestPoints(self.maps, self.size)

        mes.settingMessage("world")

        if (startingState == "c" or startingState == "center" or startingState == "centre"):
            startingState = int(((self.size)[0] * (self.size)[1]) / 2 + (self.size)[0] / 2) - 1

        self.world = pdm.stochasticMaze(self.size, self.lines, _preDefRewardSet(self.interestPoints), startingState)
        mes.setMessage("world")

        self.graph = graph

    def _initGraph(self):

        ss = (self.world)._invHashFun((self.world).currentState)
        sbs = (self.world)._invHashFun((self.agent).currentState)

        mes.settingMessage("graphic render")
        self.graphic = graphic.dispWorld(self.size, self.interestPoints, ss, sbs, self.maps)
        mes.setMessage("graphic render")

    def performAction(self, action):
        prev = (self.world)._invHashFun((self.world).currentState)

        (self.world).transition(action)

        cur = (self.world)._invHashFun((self.world).currentState)

        if (self.graph):
            (self.graphic)._moveCur(prev, cur)

    def changeBelief(self):
        prev = (self.world)._invHashFun(((self.agent).stateHistory)[(self.agent).time - 1])
        cur = (self.world)._invHashFun((self.agent).currentState)

        (self.graphic)._moveBel(prev, cur)
        (self.graphic).updateRender()

    def currentPerception(self):
        return ((self.world)._invHashFun((self.world).currentState), ((self.world).R)[(self.world).currentState])

    def _reset(self):
        prev = (self.world)._invHashFun((self.world).currentState)

        self.world._resetModel()

        cur = (self.world)._invHashFun((self.world).currentState)

        if (self.graph):
            self.changeBelief()
            (self.graphic)._moveCur(prev, cur)

            for i in range(2):
                self.graphic.invTrack()

            (self.graphic).updateRender()

    def __del__(self):
        self.lines = self.size = self.maps = 0
        print(self.__class__.__name__, "has been deleted")
