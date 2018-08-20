import secSensors as sec

def leftWall(fun, intro):
    return [fun("lWall")]

def rightWall(fun, intro):
    return [fun("rWall")]

def topWall(fun, intro):
    return [fun("tWall")]

def bottomWall(fun, intro):
    return [fun("bWall")]

def exitDetector(fun, intro):
    return [fun("isExit")]

def waterDetector(fun, intro):
    return [fun("isWater")]

def impassDetector(fun, intro):
    return [fun("isImpass")]

def foodDetector(fun, intro):
    return [fun("isFood")]

def crossRoadDetector(fun, intro):
    return [fun("isCrossRoad")]

def previousAction(fun, intro):

    if intro.actionHistory!=[]:
        action = (intro.actionHistory[-1])

        ret = [0,0,0,0]
        ret[action] = 1
    else:
        return [0,0,0,0]

    return ret