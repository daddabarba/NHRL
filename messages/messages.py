import colors as col


def message(message):
    print (message)


def currentMessage(message):
    print (col.t.HEADER + "Currently: " + col.t.ENDC + message)


def settingMessage(message):
    print (col.t.OKGREEN + "Setting: " + col.t.ENDC + message)


def setMessage(message):
    print (col.t.OKBLUE + "Set: " + col.t.ENDC + message)


def warningMessage(message):
    print (col.t.WARNING + "Warning: " + col.t.ENDC + message)


def errorMessage(message):
    print (col.t.FAIL + "Error: " + col.t.ENDC + message)
