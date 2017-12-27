import numpy as np

def _isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _convertFile(file):
    maze = open(file, "r+")

    maze.read(9)
    lines = []

    maxX = -1
    maxY = -1

    while (True):
        line = []

        for i in range(0, 4):
            maze.read(5)

            p = int(maze.read(1))
            c = maze.read(1)

            while (_isNumber(c)):
                p = p * 10 + int(c)
                c = maze.read(1)

            p = int((p - 2) / 16)
            line.append(p)

            if (not (i % 2) and p > maxX):
                maxX = p
            if ((i % 2) and p > maxY):
                maxY = p

        lines.append(line)
        c = maze.read(4)
        c = maze.read(1)

        if (c == "."):
            break
        maze.read(8)

    maze.close()

    return (lines, (maxY, maxX))


def _isHorizontal(line):
    return line[0] != line[2]


def _isVertical(line):
    return line[1] != line[3]


def _addLine(maps, line, size):
    maxX = size[1]
    maxY = size[0]

    if (_isHorizontal(line)):

        rd = line[1]
        ru = rd - 1

        for i in range(line[0], line[2]):
            if (rd < maxX):
                maps[rd][i][1] = 1
            if (ru >= 0):
                maps[ru][i][3] = 1

    elif (_isVertical(line)):

        cr = line[0]
        cl = cr - 1

        for i in range(line[1], line[3]):
            if (cr < maxY):
                maps[i][cr][2] = 1
            if (cl >= 0):
                maps[i][cl][0] = 1

    else:
        mes.errorMessage("cannot recognise wall")

    return maps


def _convertLines(lines, size):
    maxX = size[1]
    maxY = size[0]

    maps = np.zeros((maxY, maxX, 4))

    for i in range(len(lines)):
        line = lines[i]
        maps = _addLine(maps, line, size)

    return maps