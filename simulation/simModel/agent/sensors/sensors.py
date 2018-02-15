import secSensors as sec

def leftWall(fun):
    return [fun("lWall")]

def rightWall(fun):
    return [fun("rWall")]

def topWall(fun):
    return [fun("tWall")]

def bottomWall(fun):
    return [fun("bWall")]

def exitDetector(fun):
    return [fun("isExit")]

def waterDetector(fun):
    return [fun("isWater")]

def impassDetector(fun):
    return [fun("isImpass")]

def foodDetector(fun):
    return [fun("isFood")]

def crossRoadDetector(fun):
    return [fun("isCrossRoad")]