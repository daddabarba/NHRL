import secSensors as sec

def gps(fun):
    return [fun("ID")]

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