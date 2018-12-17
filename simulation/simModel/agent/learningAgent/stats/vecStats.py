import numpy as np

import copy

N = 0
MU = 1
VAR = 2

def l(v):
    return np.linalg.norm(v)

class Stats():

    def __init__(self, n=0.0, mu=0.0, var=None):

        self.data = np.empty(3, dtype=object)

        self.data[N] = n
        self.data[MU] = mu
        self.data[VAR] = var

    def __copy__(self):
        return Stats(self.data[N], copy.deepcopy(self.data[MU]), self.data[VAR])

    def reshape_mean(self):
        self.mu = np.append(self.mu, 0.0)

    def update_stats(self, newPoint, weight=1.0):

        newN = self.data[N] + weight
        newMu = (self.data[MU] * self.data[N] + newPoint * weight) / newN

        if self.data[VAR] or self.data[VAR] == 0.0:
            self.data[VAR] = (self.data[VAR] * self.data[N] + self.data[N] * ((self.data[MU] ** 2).sum()) + weight * ((newPoint ** 2).sum()) - newN * ((newMu ** 2).sum())) / newN
        else:
            self.data[VAR] = 0.0

        self.data[N] = newN
        self.data[MU] = newMu

    def scale(self, n):

        self.data[N] *= 1.0 / n
        self.data[MU] *= 1.0 / n
        self.data[VAR] *= 1.0 / (n*n)

    def getVar(self):
        return self.data[VAR]

    def getMu(self):
        return self.data[MU]

    def getN(self):
        return self.data[N]