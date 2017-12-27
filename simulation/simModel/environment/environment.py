import sys

sys.path.append('../agent/')
sys.path.append('../environment/metaModel/model/')
sys.path.append('../environment/metaModel/')

sys.path.append('../')

sys.path.append('../../GUI/')
sys.path.append('../../../messages/')

sys.path.append('../environment/features/')

import convertion as con

import agent
import modWorld as modw
import preDefModels as pdm

import features

import parameters as par

import graphic
import messages as mes

def getFeatDefinitions():
    namesList = dir(features)
    namesList = namesList[8:len(namesList)]

    functionsList = []
    for i in range(len(namesList)):
        att = getattr(features, namesList[i])
        if(callable(att)):
            functionsList.append(att);

    return tuple(functionsList)

def _getFeatures(map, size):
    definitions = getFeatDefinitions()
    ft = []

    for i in range(len(definitions)):
        ft.append([])

    for row in range(size[0]):
        for col in range(size[0]):

            for feat in range(len(definitions)):

                if definitions[feat]((row,col), map, size):
                    ft[feat].append((row,col))

    return tuple(ft)

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
