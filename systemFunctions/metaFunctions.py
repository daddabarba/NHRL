import sys

def getFunctionsDefinitions(directory):
    namesList = dir(directory)
    namesList = namesList[8:len(namesList)]

    functionsList = []
    for i in range(len(namesList)):
        att = getattr(directory, namesList[i])
        if(callable(att)):
            functionsList.append(att)

    return ( tuple(functionsList), namesList )