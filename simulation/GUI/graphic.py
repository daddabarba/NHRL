import matplotlib.pyplot as plt
import matplotlib.patches as patches

import graphicPar as par

import random

def r():
    return random.randint(0,255)

def randCol():
    return '#%02x%02x%02x' % (r(),r(),r())


class dispWorld:
    def __init__(self, size, features, startingState, startingBelievedState, maps, ftNames):
        self.track = par.startingTrack
        self.render = par.startingRender

        self.fig = plt.figure(figsize=(par.figSize, par.figSize), frameon=False)
        self.figLeg = plt.figure(figsize=(par.figSize/2, par.figSize/2), frameon=False)

        self.legend = (self.figLeg).add_subplot(111, aspect='equal')
        self.ax = (self.fig).add_subplot(111, aspect='equal')


        h = self.stateHeight = (par.totHeight - (par.border) * (size[0] + 1)) / (size[0])
        w = self.stateWidth = (par.totWidth - (par.border) * (size[1] + 1)) / (size[1])

        self.statesPatches = []
        self.colors = []

        for r in range(size[0]):
            rowPatches = []
            rowColors = []

            y = par.totHeight - ((h + par.border) * (r + 1))

            color = ((par.colorFull) if (maps[r][0][2]) else (par.colorEmpty))
            wall = patches.Rectangle((0, y), par.border, h, facecolor=color, edgecolor="none")
            (self.ax).add_patch(wall)

            for c in range(size[1]):
                x = (par.border + (w + par.border) * c)

                rect = patches.Rectangle((x, y), w, h, facecolor=par.colorEmpty, edgecolor="none")
                rowPatches.append(rect)
                (self.ax).add_patch(rect)

                rowColors.append(par.colorEmpty)

                wall1 = patches.Rectangle((x, y + h), w, par.border,
                                          facecolor=((par.colorFull) if (maps[r][c][1]) else (par.colorEmpty)),
                                          edgecolor="none")
                (self.ax).add_patch(wall1)
                wall2 = patches.Rectangle((x + w, y), par.border, h,
                                          facecolor=((par.colorFull) if (maps[r][c][0]) else (par.colorEmpty)),
                                          edgecolor="none")
                (self.ax).add_patch(wall2)

            (self.statesPatches).append(rowPatches)
            (self.colors).append(rowColors)

        y = 0

        for c in range(size[1]):
            x = (par.border + (w + par.border) * c)

            wall = wall1 = patches.Rectangle((x, y), w, par.border,
                                             facecolor=((par.colorFull) if (maps[r][c][3]) else (par.colorEmpty)),
                                             edgecolor="none")
            (self.ax).add_patch(wall)

        self.ftColors = []

        for i in range(len(features)):

            (self.ftColors).append(randCol())

            for k in range(len(features[i])):
                loc = features[i][k]

                self._replace(loc, (self.ftColors)[i])
                (self.colors)[loc[0]][loc[1]] = (self.ftColors)[i]


        y = 0.0
        x = 0.0

        h = float(1)/len(features)

        for i in range(len(features)):
            rect = patches.Rectangle((x, y), par.legW, h, facecolor=(self.ftColors)[i], edgecolor="none")
            (self.legend).add_patch(rect)

            (self.legend).text(x+par.legW +(w/4),y+(h/4),ftNames[i] + " (" + str(i) + ")", fontsize = par.fontSize)

            y += (h)

        self.cs = startingState
        self.bs = startingBelievedState

        self._replace(startingState, par.colorCurrent)
        self._replace(startingBelievedState, par.colorCurrentBel)

        plt.ion()
        plt.draw()
        plt.pause(0.01)

    def _replace(self, position, color):
        patch = (self.statesPatches)[position[0]][position[1]]
        patch.remove()

        y = 1 - (((self.stateHeight) + par.border) * ((position[0]) + 1))
        x = (par.border + ((self.stateWidth) + par.border) * (position[1]))

        newPatch = patches.Rectangle((x, y), self.stateWidth, self.stateHeight, facecolor=color, edgecolor="none")
        (self.statesPatches)[position[0]][position[1]] = newPatch

        (self.ax).add_patch(newPatch)

    def _moveCur(self, prev, cur):

        if (self.track):
            col = par.colorPrev
        else:
            col = (self.colors)[prev[0]][prev[1]]

        self._replace(prev, col)
        self._replace(cur, par.colorCurrent)

        self.cs = cur

    def _moveBel(self, prev, cur):

        if (self.track):
            col = par.colorPrevBel
        else:
            col = (self.colors)[prev[0]][prev[1]]

        self._replace(prev, col)
        self._replace(cur, par.colorCurrentBel)

        self.bs = cur

    def updateRender(self):
        if (self.render):
            plt.draw()
            plt.pause(0.1)

    def invTrack(self):
        self.track = 1 - (self.track)

        if (not (self.track)):
            for i in range(len(self.colors)):
                for k in range(len((self.colors)[0])):
                    if ((i, k) != self.cs and (i, k) != self.bs):
                        self._replace((i, k), (self.colors)[i][k])

    def cleanTrack(self):
        self.invTrack()
        self.invTrack()

    def invRender(self):
        self.render = 1 - self.render

    def __del__(self):
        print (self.__class__.__name__, "has been deleted")
