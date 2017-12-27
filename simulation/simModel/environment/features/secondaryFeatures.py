import secFeaturesPar as par

def isUpperExit(row,col,maps):
    return row == 0 and maps[row][col][1] == 0

def isLowerExit(row,col,maps,size):
    return row == (size[0] - 1) and maps[row][col][3] == 0

def isRightExit(row,col,maps,size):
    return col == (size[1] - 1) and maps[row][col][0] == 0

def isLeftExit(row,col,maps):
    return col == 0 and maps[row][col][2] == 0

def nWalls(row,col,maps):
    walls = 0
    for i in range(0, 4):
        walls += maps[row][col][i]

    return walls

def isPeriphery(row,col,size):
    leftWing = 0 <= col <= int(size[1]/par.pCP)
    rightWing = int(size[1] - (size[1]/par.pCP)) <= col <= size[1]
    upWing = 0 <= row <= int(size[0]/par.pRP)
    downWing = int(size[0] - (size[0]/par.pRP)) <= row <= size[0]

    return leftWing or rightWing or upWing or downWing

def isCentral(row,col,size):
    centralCol = int((size[1]/2) - (size[1]/par.pCC) ) <= col <= int((size[1]/2) + (size[1]/par.pCC) )
    centralRow = int((size[0] / 2) - (size[0] / par.pRC)) <= row <= int((size[0] / 2) + (size[0] / par.pRC))

    return centralCol and centralRow