import secondaryFeatures as sec
import featurePar as par

import random as rand

def isExit(location, maps, size):
    row = location[0]
    col = location[1]

    return sec.isUpperExit(row,col,maps) or sec.isLowerExit(row,col,maps,size) or sec.isRightExit(row,col,maps,size) or sec.isLeftExit(row,col,maps)

def isImpass(location, maps, size):
    row = location[0]
    col = location[1]

    walls = sec.nWalls(row,col,maps)
    return (walls>=3)

def isCrossRoad(location, maps, size):
    row = location[0]
    col = location[1]

    walls = sec.nWalls(row,col,maps)
    return (walls<=1)

def isWater(location, maps, size):
    row = location[0]
    col = location[1]

    p = float(rand.randint(0, 100)) / 100

    return sec.isPeriphery(row,col,size) and (p < par.waterP)

def isFood(location, maps, size):
    row = location[0]
    col = location[1]

    p = float(rand.randint(0, 100)) / 100

    return sec.isCentral(row,col,size) and (p < par.foodP)