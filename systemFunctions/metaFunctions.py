import inspect

def getFunctionsDefinitions(directory):
    namesList = dir(directory)
    namesList = namesList[8:len(namesList)]

    functionsList = []
    names=[]
    for i in range(len(namesList)):
        att = getattr(directory, namesList[i])
        if(callable(att)):
            functionsList.append(att)
            names.append(namesList[i])

    return ( tuple(functionsList), names )