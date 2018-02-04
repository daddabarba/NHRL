import sys

sys.path.append('../')

sys.path.append('../../../messages/')

import parameters as par

import messages as mes

_defBaseReward = -0.04
_defGoalReward = 10.0

_defSA = 10.0
_defSB = -1.0
_defSThresh = 0.001

_defIA = 100.0
_defIB = 5.0
_defIC = 1.0
_defILowBound = 0.0

_defLearningRate = 0.2
_defDiscountFactor = 0.98

_defStartQMin = 2 * par.baseReward * (-1 if par.baseReward > 0 else 1)
_defStartQMax = _defStartQMin * (-1)


def defInput(mes, defVal):
    return float(input("(" + str(defVal) + ") - " + mes) or str(defVal))


class agentPar:
    def __init__(self):
        mes.currentMessage("Setting base reward value to " + str(_defBaseReward))
        self.baseReward = _defBaseReward

        mes.currentMessage("Setting goal reward value to " + str(_defGoalReward))
        self.goalReward = _defGoalReward

        mes.currentMessage("Setting parameter A of scheduling function to " + str(_defSA))
        self.scheduleA = _defSA

        mes.currentMessage("Setting parameter B of scheduling function to " + str(_defSB))
        self.scheduleB = _defSB

        mes.currentMessage("Setting threshold of scheduling function to " + str(_defSThresh))
        self.scheduleThresh = _defSThresh

        mes.currentMessage("Setting parameter A of interest update rule to " + str(_defIA))
        self.interestA = _defIA

        mes.currentMessage("Setting parameter B of interest update rule to " + str(_defIB))
        self.interestB = _defIB

        mes.currentMessage("Setting parameter C of interest update rule to " + str(_defIC))
        self.interestC = _defIC

        mes.currentMessage("Setting lower bound parameter of interest update rule to " + str(_defILowBound))
        self.iLowBound = _defILowBound

        mes.currentMessage("Setting learning rate to: " + str(_defLearningRate))
        self.learningRate = _defLearningRate

        mes.currentMessage("Setting discount factor to: " + str(_defDiscountFactor))
        self.discountFactor = _defDiscountFactor

        mes.currentMessage("Setting maximum starting Q value to " + str(_defStartQMax))
        self.startQMax = _defStartQMax

        mes.currentMessage("Setting minimum starting Q value to " + str(_defStartQMin))
        self.startQMin = _defStartQMin

    def printPars(self):

        print("\nReward values:")
        print("\tbase reward: " + str(self.baseReward))
        print("\tgoal reward: " + str(self.goalReward))

        print("\nScheduling function:")
        print("\tParameter A value of scheduling function: " + str(self.scheduleA))
        print("\tParameter B value of scheduling function: " + str(self.scheduleB))
        print("\tThreshold value of scheduling function: " + str(self.scheduleThresh))

        print("\nQ learning:")
        print("\tLearning rate value: " + str(self.learningRate))
        print("\tDiscount factor value: " + str(self.discountFactor))
        print("\tMaximum starting Q value: " + str(self.startQMax))
        print("\tMinimum starting Q value: " + str(self.startQMin))

        print("\nInterest value update rule")
        print("\tParameter A value of interest update rule: " + str(self.interestA))
        print("\tParameter B value of interest update rule: " + str(self.interestB))
        print("\tParameter C value of interest update rule: " + str(self.interestC))
        print("\tLow bound value of interest update rule: " + str(self.iLowBound))

        print("\n")

    def resetPars(self):
        self.baseReward = _defBaseReward
        self.goalReward = _defGoalReward

        self.scheduleA = _defSA
        self.scheduleB = _defSB
        self.scheduleThresh = _defSThresh

        self.interestA = _defIA
        self.interestB = _defIB
        self.interestC = _defIC
        self.iLowBound = _defILowBound

        self.learningRate = _defLearningRate
        self.discountFactor = _defDiscountFactor
        self.startQMin = _defStartQMin
        self.startQMax = _defStartQMax

    def changePars(self):
        self.baseReward = defInput("Insert base reward value: ", self.baseReward)
        self.goalReward = defInput("Insert goal reward value: ", self.goalReward)

        self.scheduleA = defInput("Insert parameter A of scheduling function: ", self.scheduleA)
        self.scheduleB = defInput("Insert parameter B of scheduling function: ", self.scheduleB)
        self.scheduleThresh = defInput("Insert threshold of scheduling function: ", self.scheduleThresh)

        self.interestA = defInput("Insert parameter A of interest value update rule: ", self.interestA)
        self.interestB = defInput("Insert parameter B of interest value update rule: ", self.interestB)
        self.interestC = defInput("Insert parameter C of interest value update rule: ", self.interestC)
        self.iLowBound = defInput("Insert low bound parameter of interest value update rule: ", self.iLowBound)

        self.learningRate = defInput("Insert learning rate value: ", self.learningRate)
        self.discountFactor = defInput("Insert discount factor value: ", self.discountFactor)
        self.startQMax = defInput("Insert maximum starting Q value: ", self.startQMax)
        self.startQMin = defInput("Insert minimum starting Q value: ", self.startQMin)

    def __del__(self):
        self.baseReward = self.goalReward = self.scheduleA = self.scheduleB = self.scheduleThresh = self.interestA = self.interestB = self.interestC = self.iLowBound = self.learningRate = self.discountFactor = self.startQMax = self.startQMin = 0
        print (self.__class__.__name__, "has been deleted")
