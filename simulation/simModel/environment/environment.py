import sys

sys.path.append('../agent/')
sys.path.append('../environment/metaModel/model/')
sys.path.append('../environment/metaModel/')

sys.path.append('../')

sys.path.append('../../GUI/')
sys.path.append('../../../messages/')

import numpy as np
import random as rand

import convertion as con

import agent
import modWorld as modw
import preDefModels as pdm

import parameters as par

import graphic
import messages as mes



def _getFeatures(maps, size):
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


def _preDefRewardSet(features):
    rewardSet = []

    for k in range(len(features)):

        signalSet = [(par.baseReward)]

        for i in range(len(features[k])):
            signalSet.append((features[k][i], par.signalReward))

        rewardSet.append(signalSet)

    return rewardSet


class environment:
    def __init__(self, fileName, agent, startingState, graph):
        mes.currentMessage("linking agent")
        (self.agent) = agent
        mes.currentMessage("converting SVG file into lines format")
        (self.lines, self.size) = con._convertFile(fileName)
        mes.currentMessage("converting lines format into map format")
        (self.maps) = con._convertLines(self.lines, self.size)

        mes.currentMessage("retreiving interest points")
        self.features = _getFeatures(self.maps, self.size)

        mes.settingMessage("world")

        if (startingState == "c" or startingState == "center" or startingState == "centre"):
            startingState = int(((self.size)[0] * (self.size)[1]) / 2 + (self.size)[0] / 2) - 1

        self.world = pdm.stochasticMaze(self.size, self.lines, _preDefRewardSet(self.features), startingState)
        mes.setMessage("world")

        self.graph = graph

    def _initGraph(self):

        ss = (self.world)._invHashFun((self.world).currentState)
        sbs = (self.world)._invHashFun((self.agent).currentState)

        mes.settingMessage("graphic render")
        self.graphic = graphic.dispWorld(self.size, self.features, ss, sbs, self.maps)
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
