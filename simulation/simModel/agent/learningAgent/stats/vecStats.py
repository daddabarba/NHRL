import numpy as np

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