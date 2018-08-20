import colors as col

suppress = False

def message(message):
    if not suppress:
        print (message)


def currentMessage(message):
    if not suppress:
        print (col.t.HEADER + "Currently: " + col.t.ENDC + message)


def settingMessage(message):
    if not suppress:
        print (col.t.OKGREEN + "Setting: " + col.t.ENDC + message)


def setMessage(message):
    if not suppress:
        print (col.t.OKBLUE + "Set: " + col.t.ENDC + message)


def warningMessage(message):
    if not suppress:
        print (col.t.WARNING + "Warning: " + col.t.ENDC + message)


def errorMessage(message):
    if not suppress:
        print (col.t.FAIL + "Error: " + col.t.ENDC + message)
